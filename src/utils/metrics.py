"""
metrics.py

Single-label, 4-class evaluation metrics for the neonatal resuscitation activity
recognition task. This is the sole metric-computation point shared by
training.py and test.py, mirroring the multimodal thesis's video base model
(MoViNet) so the numbers are directly comparable.

Classes (index order = Repo 1 `label_num`):
    0 non_target, 1 stimulation, 2 ventilation, 3 suction

Difference vs. the multimodal repo's metrics.py:
    * Predictions come from argmax over the 4 softmax logits (single label),
      NOT per-class sigmoid thresholds. There are no decision thresholds.
    * Reports per-class precision/recall/F1, macro averages, plain accuracy,
      the minority-class F1 (the thesis tracked "best minority-class F1"), and
      a flattened confusion matrix.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

CLASSES = ["non_target", "stimulation", "ventilation", "suction"]
CLASS_INDICES = list(range(len(CLASSES)))
# The rare target class the thesis tracked a dedicated checkpoint for.
DEFAULT_MINORITY_CLASS = "suction"


def compute_metrics(logits, labels, minority_class: str = DEFAULT_MINORITY_CLASS):
    """
    Args:
        logits (Tensor): (N, 4) raw pre-softmax logits (any device).
        labels (Tensor): (N,) int64 ground-truth class indices, OR (N, 4)
                         one-hot; both are handled.
        minority_class (str): class name whose F1 is surfaced as
                              "minority/f1" (default "suction").
    Returns:
        dict[str, float]: per-class + macro + accuracy + minority F1, plus a
        flattened confusion matrix under keys "cm/{true}->{pred}".
    """
    logits = logits.detach().cpu()
    y_pred = torch.argmax(logits, dim=1).numpy()

    labels = labels.detach().cpu()
    if labels.ndim == 2:  # one-hot -> indices
        y_true = torch.argmax(labels, dim=1).numpy()
    else:
        y_true = labels.long().numpy()

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_INDICES, average=None, zero_division=0
    )
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_INDICES, average="macro", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)

    metrics = {f"{CLASSES[i]}/precision": float(prec[i]) for i in CLASS_INDICES}
    metrics.update({f"{CLASSES[i]}/recall": float(rec[i]) for i in CLASS_INDICES})
    metrics.update({f"{CLASSES[i]}/f1": float(f1[i]) for i in CLASS_INDICES})
    metrics.update({
        "macro/precision": float(prec_m),
        "macro/recall": float(rec_m),
        "macro/f1": float(f1_m),
        "macro/accuracy": float(acc),
    })

    if minority_class in CLASSES:
        metrics["minority/f1"] = float(f1[CLASSES.index(minority_class)])

    cm = confusion_matrix(y_true, y_pred, labels=CLASS_INDICES)
    for i in CLASS_INDICES:
        for j in CLASS_INDICES:
            metrics[f"cm/{CLASSES[i]}->{CLASSES[j]}"] = int(cm[i, j])

    return metrics
