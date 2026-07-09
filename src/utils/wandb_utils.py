"""
wandb_utils.py

Thin, optional Weights & Biases helpers shared by training.py and test.py.

All functions are no-ops when wandb is not installed or no run is active, so the
pipeline runs unchanged with `wandb_mode: disabled` in the config. Metric dicts
from compute_metrics() carry both scalar entries ("macro/f1", "suction/f1", ...)
and flattened confusion-matrix counts ("cm/<true>-><pred>"); the scalar/plot
split is handled here so callers stay clean.
"""

from __future__ import annotations

from .metrics import CLASSES

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


def log_confusion_matrix(logits, labels, key: str, extra: dict | None = None) -> None:
    """
    Log a wandb confusion-matrix plot from raw logits (N,C) and ground-truth
    class indices (N,). Predictions are argmax over the logits.
    """
    if not available():
        return
    import torch  # local import: only needed when actually logging

    y_pred = torch.as_tensor(logits).detach().cpu().argmax(dim=1).numpy().astype(int)
    lab = torch.as_tensor(labels).detach().cpu()
    y_true = (lab.argmax(dim=1) if lab.ndim == 2 else lab).numpy().astype(int)

    payload = {
        key: wandb.plot.confusion_matrix(
            y_true=y_true.tolist(),
            preds=y_pred.tolist(),
            class_names=CLASSES,
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
