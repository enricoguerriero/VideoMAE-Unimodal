"""
metrics.py

Evaluation metrics for the neonatal resuscitation activity recognition task, for
EITHER task defined by configs/data.yaml. This is the sole metric-computation
point shared by training.py, test.py and infer_video.py.

    multiclass : predictions are argmax over the softmax logits. Per-class
                 precision/recall/F1, macro averages, plain accuracy, the
                 minority-class F1, and a flattened NxN confusion matrix.
                 Mirrors the multimodal thesis' MoViNet video base model, so the
                 numbers are directly comparable.

    multilabel : predictions are independent per-activity sigmoid cuts at
                 `decision_thresholds`. Per-activity precision/recall/F1 plus
                 threshold-free average precision, macro averages, exact-match
                 and Hamming accuracy, per-activity 2x2 confusion counts — and a
                 PROJECTED single-label confusion matrix (see `project_to_single`)
                 so the thesis' 4-class table can still be reported.

Whatever the task, the following keys always exist, so callers never branch:
    macro/f1, macro/precision, macro/recall, macro/accuracy, minority/f1

`macro/accuracy` means plain accuracy in multiclass and EXACT-MATCH (subset)
accuracy in multilabel — the strictest reading, and the one that degrades if any
single activity is wrong.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

#: Fallback when a checkpoint/config names no minority class.
DEFAULT_MINORITY_CLASS = "suction"


# ---------------------------------------------------------------------------
# Probabilities and hard predictions
# ---------------------------------------------------------------------------
def to_probs(logits, spec):
    """Raw logits -> probabilities, using the spec's output activation.

    softmax over classes (multiclass) or independent sigmoids (multilabel).
    """
    logits = torch.as_tensor(logits).detach().float()
    if spec.is_multilabel:
        return torch.sigmoid(logits)
    return torch.softmax(logits, dim=-1)


def hard_predictions(probs, spec):
    """Probabilities -> hard predictions.

    multiclass: (N,) int64 class indices (argmax).
    multilabel: (N, C) int64 0/1 matrix (per-activity threshold).
    """
    probs = torch.as_tensor(probs).detach().float()
    if spec.is_multilabel:
        thr = torch.tensor(spec.sigmoid_thresholds(), dtype=probs.dtype)
        return (probs >= thr).long()
    return probs.argmax(dim=-1)


def project_to_single(probs, spec):
    """Multilabel probabilities -> single-label class index, for the projected
    (thesis-comparable) confusion matrix.

    Rule: an activity is active when it clears its own threshold; the prediction
    is the highest-probability active activity, or `negative_class` (index 0) when
    none is active. Note this discards genuine co-occurrence — it exists purely
    to keep one table comparable with the 4-class single-label results.
    """
    probs = torch.as_tensor(probs).detach().float().numpy()
    thr = np.asarray(spec.sigmoid_thresholds())
    active = probs >= thr
    out = np.zeros(len(probs), dtype=int)
    for i in range(len(probs)):
        if active[i].any():
            masked = np.where(active[i], probs[i], -np.inf)
            out[i] = 1 + int(np.argmax(masked))
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(logits, labels, spec, masks=None, minority_class=None):
    """
    Args:
        logits (Tensor): (N, C) raw pre-activation logits (any device).
        labels (Tensor): (N,) int64 class indices (multiclass) or (N, C) float
                         0/1 targets (multilabel).
        spec (DataSpec): the loaded data config — decides everything below.
        masks (Tensor|None): (N, C) 1 = supervised, 0 = excluded. Multilabel
                         only; masked elements are left out of every figure.
        minority_class (str|None): class whose F1 is surfaced as "minority/f1".

    Returns:
        dict[str, float|int]: scalar metrics plus flattened confusion-matrix
        counts under "cm/..." keys.
    """
    minority_class = minority_class or DEFAULT_MINORITY_CLASS
    probs = to_probs(logits, spec)
    if spec.is_multilabel:
        return _multilabel_metrics(probs, labels, spec, masks, minority_class)
    return _multiclass_metrics(probs, labels, spec, minority_class)


def _multiclass_metrics(probs, labels, spec, minority_class):
    classes = spec.class_names
    idx = list(range(len(classes)))
    y_pred = hard_predictions(probs, spec).numpy()

    labels = torch.as_tensor(labels).detach().cpu()
    y_true = (labels.argmax(dim=1) if labels.ndim == 2 else labels.long()).numpy()

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=idx, average=None, zero_division=0)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=idx, average="macro", zero_division=0)

    metrics = {}
    for i in idx:
        metrics[f"{classes[i]}/precision"] = float(prec[i])
        metrics[f"{classes[i]}/recall"] = float(rec[i])
        metrics[f"{classes[i]}/f1"] = float(f1[i])
    metrics.update({
        "macro/precision": float(prec_m),
        "macro/recall": float(rec_m),
        "macro/f1": float(f1_m),
        "macro/accuracy": float(accuracy_score(y_true, y_pred)),
    })
    if minority_class in classes:
        metrics["minority/f1"] = float(f1[classes.index(minority_class)])

    cm = confusion_matrix(y_true, y_pred, labels=idx)
    for i in idx:
        for j in idx:
            metrics[f"cm/{classes[i]}->{classes[j]}"] = int(cm[i, j])
    return metrics


def _multilabel_metrics(probs, labels, spec, masks, minority_class):
    activities = list(spec.activities)
    y_score = probs.numpy()
    y_pred = hard_predictions(probs, spec).numpy()
    y_true = torch.as_tensor(labels).detach().cpu().float().numpy()
    y_true = (y_true >= 0.5).astype(int)

    if masks is None:
        m = np.ones_like(y_true, dtype=bool)
    else:
        m = torch.as_tensor(masks).detach().cpu().numpy() >= 0.5

    metrics = {}
    precs, recs, f1s, aps = [], [], [], []
    for c, name in enumerate(activities):
        sel = m[:, c]
        t, p, s = y_true[sel, c], y_pred[sel, c], y_score[sel, c]
        if sel.sum() == 0:
            # Every clip in this split is ambiguous for this activity, so it has
            # no supervised examples at all. Report zeros rather than letting
            # sklearn raise on the empty arrays — this runs every epoch, and
            # `<name>/n_supervised = 0` below is what makes the cause visible.
            prec = rec = f1 = 0.0
        else:
            prec, rec, f1, _ = precision_recall_fscore_support(
                t, p, labels=[1], average="binary", pos_label=1, zero_division=0)
        # Average precision is undefined without a positive; report 0 and let
        # the support counts below explain why.
        ap = float(average_precision_score(t, s)) if t.sum() > 0 else 0.0
        tn = int(((t == 0) & (p == 0)).sum())
        fp = int(((t == 0) & (p == 1)).sum())
        fn = int(((t == 1) & (p == 0)).sum())
        tp = int(((t == 1) & (p == 1)).sum())

        metrics[f"{name}/precision"] = float(prec)
        metrics[f"{name}/recall"] = float(rec)
        metrics[f"{name}/f1"] = float(f1)
        metrics[f"{name}/ap"] = ap
        metrics[f"{name}/support"] = int(t.sum())
        metrics[f"{name}/n_supervised"] = int(sel.sum())
        metrics[f"cm/{name}/tn"] = tn
        metrics[f"cm/{name}/fp"] = fp
        metrics[f"cm/{name}/fn"] = fn
        metrics[f"cm/{name}/tp"] = tp
        precs.append(float(prec)); recs.append(float(rec))
        f1s.append(float(f1)); aps.append(ap)

    fully = m.all(axis=1)
    exact = float((y_true[fully] == y_pred[fully]).all(axis=1).mean()) if fully.any() else 0.0
    hamming = float((y_true[m] == y_pred[m]).mean()) if m.any() else 0.0

    metrics.update({
        "macro/precision": float(np.mean(precs)),
        "macro/recall": float(np.mean(recs)),
        "macro/f1": float(np.mean(f1s)),
        "macro/ap": float(np.mean(aps)),
        "macro/accuracy": exact,      # exact-match (subset) accuracy
        "hamming/accuracy": hamming,
        "n_fully_supervised": int(fully.sum()),
    })
    if minority_class in activities:
        metrics["minority/f1"] = float(f1s[activities.index(minority_class)])
        metrics["minority/ap"] = float(aps[activities.index(minority_class)])
    else:
        metrics["minority/f1"] = 0.0

    # Co-occurrence: how much multi-activity truth exists, and did we find it?
    metrics["true/multi_active"] = int((y_true.sum(axis=1) >= 2).sum())
    metrics["pred/multi_active"] = int((y_pred.sum(axis=1) >= 2).sum())

    # ---- projected single-label view (thesis-comparable) ------------------
    y_true_proj, keep = _project_targets(y_true, fully)
    if keep.any():
        y_pred_proj = project_to_single(probs, spec)[keep]
        names = spec.projected_class_names
        idx = list(range(len(names)))
        _, _, pf1, _ = precision_recall_fscore_support(
            y_true_proj, y_pred_proj, labels=idx, average="macro", zero_division=0)
        metrics["proj/macro_f1"] = float(pf1)
        metrics["proj/accuracy"] = float(accuracy_score(y_true_proj, y_pred_proj))
        cm = confusion_matrix(y_true_proj, y_pred_proj, labels=idx)
        for i in idx:
            for j in idx:
                metrics[f"cm/{names[i]}->{names[j]}"] = int(cm[i, j])
    metrics["proj/excluded"] = int((~keep).sum())
    return metrics


def _project_targets(y_true, fully_supervised):
    """Ground-truth multilabel -> single-label index, plus a keep mask.

    Clips with >= 2 true activities have no honest single-label answer, and
    partially-masked clips have no complete one; both are EXCLUDED rather than
    resolved by a tie-break that would flatter the model. `proj/excluded`
    reports how many.
    """
    n_active = y_true.sum(axis=1)
    keep = fully_supervised & (n_active <= 1)
    sub = y_true[keep]
    out = np.zeros(len(sub), dtype=int)
    rows, cols = np.nonzero(sub)
    out[rows] = cols + 1
    return out, keep


def single_label_view(logits, labels, spec, masks=None):
    """(y_true, y_pred, class_names) as single-label integer arrays.

    multiclass: argmax vs the true index. multilabel: the projected view, with
    ambiguous ground truth excluded. Used for the W&B confusion-matrix plot,
    which needs one label per sample in both tasks.
    """
    probs = to_probs(logits, spec)
    if not spec.is_multilabel:
        labels = torch.as_tensor(labels).detach().cpu()
        y_true = (labels.argmax(dim=1) if labels.ndim == 2 else labels.long()).numpy()
        return y_true, hard_predictions(probs, spec).numpy(), spec.class_names

    y_true_raw = (torch.as_tensor(labels).detach().cpu().float().numpy() >= 0.5).astype(int)
    if masks is None:
        fully = np.ones(len(y_true_raw), dtype=bool)
    else:
        fully = (torch.as_tensor(masks).detach().cpu().numpy() >= 0.5).all(axis=1)
    y_true, keep = _project_targets(y_true_raw, fully)
    return y_true, project_to_single(probs, spec)[keep], spec.projected_class_names
