"""
losses.py

Task-driven loss construction. The DataSpec decides which loss the run uses:

    multiclass -> CrossEntropyLoss(weight=sqrt inverse-frequency class weights,
                  label_smoothing=...)                      [the thesis' loss]
    multilabel -> MaskedBCEWithLogitsLoss(pos_weight=sqrt(neg/pos) per activity,
                  label_smoothing=...)

Both are wrapped so training.py and test.py can call them identically:

    loss = criterion(logits, labels, mask)      # mask is None in multiclass

That uniform signature is the whole point — no `if task == ...` branches in the
training loop.
"""

import torch
import torch.nn as nn


class MaskedBCEWithLogitsLoss(nn.Module):
    """BCEWithLogitsLoss over independent per-activity sigmoids, with a
    per-element supervision mask.

    The mask is what makes the ambiguous band usable. A clip covering 40 %
    stimulation and 60 % ventilation is a confident ventilation POSITIVE and a
    genuinely unknown stimulation label; masking the stimulation term keeps the
    clip's usable supervision without inventing a negative. See
    `ambiguous: mask` in configs/data.yaml.

    Reduction is the mean over SUPERVISED elements only, so the loss scale does
    not drift as the number of masked elements changes across batches.
    """

    def __init__(self, pos_weight=None, label_smoothing: float = 0.0):
        super().__init__()
        if not 0.0 <= label_smoothing < 0.5:
            raise ValueError(f"label_smoothing must be in [0, 0.5), got {label_smoothing}")
        self.label_smoothing = float(label_smoothing)
        self.register_buffer(
            "pos_weight",
            None if pos_weight is None else torch.as_tensor(pos_weight, dtype=torch.float32),
        )

    def forward(self, logits, targets, mask=None):
        targets = targets.to(dtype=logits.dtype)
        if self.label_smoothing:
            # Binary label smoothing: pull both 0 and 1 toward 0.5.
            eps = self.label_smoothing
            targets = targets * (1.0 - eps) + 0.5 * eps
        pw = None if self.pos_weight is None else self.pos_weight.to(logits.dtype)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw, reduction="none")
        if mask is None:
            return loss.mean()
        mask = mask.to(dtype=loss.dtype)
        return (loss * mask).sum() / mask.sum().clamp(min=1.0)


class _CrossEntropyAdapter(nn.Module):
    """CrossEntropyLoss with the (logits, labels, mask) signature. `mask` is
    accepted and ignored — a softmax over mutually-exclusive classes has no
    per-class supervision to switch off."""

    def __init__(self, weight=None, label_smoothing: float = 0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

    def forward(self, logits, targets, mask=None):
        return self.ce(logits, targets)


def build_criterion(spec, dataset, config, device):
    """Build the loss for `spec`, with weights derived from `dataset`.

    Args:
        spec (DataSpec): the loaded data config.
        dataset (VideoMAEDataset): the TRAIN split — class statistics come from
            here, never from validation or test.
        config (dict): configs/config.yaml. Reads `label_smoothing` and
            `class_weighting` ("sqrt_inv_freq" | "inv_freq" | "none").
        device: where to place the weight tensors.

    Returns:
        (criterion, weights) — `weights` is the tensor that was applied
        (CE `weight` or BCE `pos_weight`), for logging.
    """
    smoothing = float(config.get("label_smoothing", 0.0))
    weighting = str(config.get("class_weighting", "sqrt_inv_freq"))

    if spec.is_multilabel:
        pos_weight = dataset.compute_pos_weight(weighting).to(device)
        return MaskedBCEWithLogitsLoss(pos_weight=pos_weight,
                                       label_smoothing=smoothing).to(device), pos_weight

    class_weights = dataset.compute_class_weights(weighting).to(device)
    return _CrossEntropyAdapter(weight=class_weights,
                                label_smoothing=smoothing).to(device), class_weights
