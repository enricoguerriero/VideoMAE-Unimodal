"""
wandb_utils.py

Thin, optional Weights & Biases helpers shared by training.py and test.py.

All functions are no-ops when wandb is not installed or no run is active, so the
pipeline runs unchanged with `wandb_mode: disabled` in the config. Metric dicts
from compute_metrics() carry both scalar entries ("macro/f1", "suction/f1", ...)
and flattened confusion-matrix counts ("cm/..."); the scalar/plot split is
handled here so callers stay clean.

Nothing here knows about the task. Class names and the single-label view needed
for the confusion-matrix plot are supplied by the caller via the DataSpec, so the
same helpers serve both multiclass and multilabel runs.
"""

from __future__ import annotations

from .metrics import single_label_view

try:
    import wandb
    _HAS_WANDB = True
except Exception:  # pragma: no cover
    _HAS_WANDB = False


def available() -> bool:
    """True if wandb is importable AND a run has been initialised."""
    return _HAS_WANDB and wandb.run is not None


def scalar_metrics(metrics: dict) -> dict:
    """Drop the flattened confusion-matrix ('cm/...') keys, keep scalars."""
    return {k: v for k, v in metrics.items() if not k.startswith("cm/")}


def define_epoch_metrics() -> None:
    """
    Make 'epoch' the x-axis for all val/*, train/loss_epoch, and lr charts, and
    'train/global_step' the x-axis for the per-step training loss. Call once,
    right after wandb.init.
    """
    if not available():
        return
    wandb.define_metric("train/global_step")
    wandb.define_metric("epoch")
    wandb.define_metric("train/loss", step_metric="train/global_step")
    wandb.define_metric("train/loss_epoch", step_metric="epoch")
    wandb.define_metric("lr", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("val_step/*", step_metric="train/global_step")


def log(payload: dict) -> None:
    """wandb.log wrapper that is a no-op when unavailable."""
    if available():
        wandb.log(payload)


def log_metrics(metrics: dict, prefix: str, extra: dict | None = None) -> None:
    """Log the scalar part of a metrics dict under `prefix` (e.g. 'val/')."""
    if not available():
        return
    payload = {f"{prefix}{k}": float(v) for k, v in scalar_metrics(metrics).items()}
    if extra:
        payload.update(extra)
    wandb.log(payload)


def log_confusion_matrix(logits, labels, spec, key: str, masks=None,
                         extra: dict | None = None) -> None:
    """Log a wandb confusion-matrix plot from raw logits and ground truth.

    multiclass: the NxN argmax-vs-truth matrix.
    multilabel: the PROJECTED single-label matrix (see metrics.project_to_single)
                — a plot needs one label per sample, and this keeps the figure
                comparable with the thesis' 4-class table. Clips with >= 2 true
                activities are excluded from it; the "proj/excluded" metric
                reports how many.
    """
    if not available():
        return
    y_true, y_pred, class_names = single_label_view(logits, labels, spec, masks)
    if len(y_true) == 0:
        return
    payload = {
        key: wandb.plot.confusion_matrix(
            y_true=[int(v) for v in y_true],
            preds=[int(v) for v in y_pred],
            class_names=list(class_names),
        )
    }
    if extra:
        payload.update(extra)
    wandb.log(payload)


def update_summary(values: dict) -> None:
    """Write final/best values to the run summary (shown in the runs table)."""
    if not available():
        return
    for k, v in values.items():
        wandb.run.summary[k] = v


def log_artifact(name: str, artifact_type: str, files: list[str], metadata: dict | None = None) -> None:
    """Store output files (e.g. scores.npz, results.csv) as a wandb Artifact."""
    if not available():
        return
    artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata or {})
    for f in files:
        artifact.add_file(f)
    wandb.log_artifact(artifact)


def finish() -> None:
    if available():
        wandb.finish()
