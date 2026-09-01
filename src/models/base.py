"""
base.py

VideoModel — abstract base class for the VideoMAE backbones in this repo.

This is a trimmed, self-contained version of the multimodal repo's
`VisionLanguageModel`: all VLM/LoRA/PEFT machinery has been removed because this
project only fine-tunes pure vision transformers (VideoMAE-base, VideoMAEv2-giant)
with no language stream. Removing PEFT also removes a heavy dependency, keeping
the repo standalone.

TASK-DRIVEN OUTPUT LAYER
------------------------
`num_classes` and `task` both come from configs/data.yaml via DataSpec, so the
head width and its output activation follow the data:

    multiclass -> 1 + len(activities) logits, SOFTMAX  (mutually exclusive)
    multilabel ->     len(activities) logits, SIGMOID  (independent activities)

`forward()` always returns RAW LOGITS — both CrossEntropyLoss and
BCEWithLogitsLoss want pre-activation values, and fusing the activation into the
graph would cost numerical stability for nothing. Anything that needs actual
probabilities calls `model.probs(logits)`, which applies the activation the task
requires. `model.task` / `model.activation` say which one that is.

Shared responsibilities kept here:
    * classifier-head construction (build_classifier)
    * optional attention pooling (build_attention_pooling)
    * checkpoint restore helpers (load_backbone / load_classifier /
      load_attention_pooling)

Concrete subclasses (videomae.py, videomae_giant.py) implement only their
backbone-specific __init__ and forward.
"""

import logging

import torch
import torch.nn as nn

from .classifier import ClassifierHead
from .attentionpooling import AttentionPooling

logger = logging.getLogger(__name__)

MULTICLASS = "multiclass"
MULTILABEL = "multilabel"


class VideoModel(nn.Module):
    def __init__(self, num_classes: int = 4, backbone_id: str = None, device=None,
                 task: str = MULTICLASS):
        """
        Args:
            num_classes (int): number of output logits. Comes from
                               DataSpec.num_classes — 1 + len(activities) in
                               multiclass, len(activities) in multilabel.
            backbone_id (str|None): HF model id, stored for reference.
            device (str|None): device string, stored for reference; concrete
                               placement handled by each subclass __init__.
            task (str): "multiclass" (softmax) or "multilabel" (sigmoid). Only
                        affects `probs()` and the head's bias semantics; the
                        forward pass returns logits either way.
        """
        super().__init__()
        if task not in (MULTICLASS, MULTILABEL):
            raise ValueError(f"task must be {MULTICLASS!r} or {MULTILABEL!r}, got {task!r}")
        self.device = device
        self.model_name = "VideoModel"
        self.num_classes = num_classes
        self.task = task
        self.backbone_id = backbone_id
        self.backbone = None
        self.processor = None
        self.hidden_size = None
        self.classifier = None
        self.attn_pool = None
        self.input_device = None

    # ------------------------------------------------------------------
    # Output activation
    # ------------------------------------------------------------------
    @property
    def is_multilabel(self) -> bool:
        return self.task == MULTILABEL

    @property
    def activation(self) -> str:
        """The output activation this task uses: 'sigmoid' or 'softmax'."""
        return "sigmoid" if self.is_multilabel else "softmax"

    def probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply the task's output activation to raw logits.

        multilabel: independent per-activity sigmoids — the rows do NOT sum to 1,
                    which is exactly what makes co-occurrence representable.
        multiclass: softmax over mutually exclusive classes.
        """
        return torch.sigmoid(logits) if self.is_multilabel else torch.softmax(logits, dim=-1)

    # ------------------------------------------------------------------
    # Head / pooling factories
    # ------------------------------------------------------------------
    def build_classifier(self, classifier_config: dict, bias=None):
        """Construct and register the ClassifierHead from config."""
        self.classifier = ClassifierHead(
            in_dim=self.hidden_size,
            dims=classifier_config.get("dims", []),
            num_classes=self.num_classes,
            activation=classifier_config.get("activation", "relu"),
            dropout=classifier_config.get("classifier_dropout", classifier_config.get("dropout", 0.2)),
            bias=bias if classifier_config.get("use_bias", True) else None,
        )

    def build_attention_pooling(self):
        """Construct and register an AttentionPooling module."""
        self.attn_pool = AttentionPooling(self.hidden_size)

    # ------------------------------------------------------------------
    # Checkpoint restore
    # ------------------------------------------------------------------
    def load_backbone(self, checkpoint: dict, config: dict = None):
        """Restore backbone weights (no LoRA). Subclasses may override.

        strict=False is deliberate here (checkpoints predating `fc_norm` must
        still load), but anything skipped is REPORTED — a silently half-restored
        backbone is indistinguishable from a badly trained one.
        """
        missing, unexpected = self.backbone.load_state_dict(
            checkpoint["backbone"], strict=False)
        if missing or unexpected:
            logger.warning(f"backbone restore skipped keys — missing {list(missing)[:8]}"
                           f"{'...' if len(missing) > 8 else ''}, unexpected "
                           f"{list(unexpected)[:8]}{'...' if len(unexpected) > 8 else ''}")

    def load_classifier(self, checkpoint: dict, config: dict = None):
        """Build the head from config, restore its weights, move to backbone device.

        The head's OUTPUT bias is CONSTRUCTED CONDITIONALLY — ClassifierHead
        builds its last layer as `nn.Linear(..., bias=(bias is not None))` — so the
        head must be rebuilt with a bias whenever the checkpoint has one. Building it without and
        relying on `strict=False` silently drops the trained bias: the weights
        still load, so the ranking (and average precision) look perfect, while
        every logit shifts by the bias that went missing. For a rare activity
        that bias is a large negative number — the logit prior, e.g.
        ln(0.05/0.95) = -2.9 — so dropping it pushes the sigmoid over 0.5
        almost everywhere and F1 collapses to 2p/(1+p) while AP stays high.
        """
        classifier_config = (config or {}).get("classifier_config", {})
        saved = checkpoint["classifier"]

        # Locate the OUTPUT layer explicitly (highest `seq.<i>` index) rather than
        # trusting state_dict ordering, and read its bias — not "any bias anywhere
        # in the head". Hidden layers always carry a bias, so `any(...bias)` would
        # claim an output bias for a `use_bias: false` MLP and rebuild the wrong
        # architecture.
        idxs = [int(k.split(".")[1]) for k in saved
                if len(k.split(".")) > 2 and k.split(".")[1].isdigit()]
        last = max(idxs, default=None)
        out_w = saved.get(f"seq.{last}.weight") if last is not None else None
        if out_w is not None and out_w.shape[0] != self.num_classes:
            raise ValueError(
                f"checkpoint head emits {out_w.shape[0]} logits but this run's "
                f"data config asks for {self.num_classes} ({self.task}). The "
                f"checkpoint was trained for a different task — load the matching "
                f"data config (checkpoints store theirs under 'data_spec').")

        # Mirror the saved head's shape: a placeholder bias vector is enough,
        # since load_state_dict overwrites it with the trained values.
        has_out_bias = last is not None and f"seq.{last}.bias" in saved
        placeholder = torch.zeros(self.num_classes) if has_out_bias else None
        self.build_classifier(classifier_config, bias=placeholder)

        missing, unexpected = self.classifier.load_state_dict(saved, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"classifier head does not match the checkpoint — missing "
                f"{list(missing)}, unexpected {list(unexpected)}. Loading it "
                f"anyway would silently change what the logits mean; this used to "
                f"be swallowed by strict=False. Check `classifier_config` "
                f"(dims/use_bias) against the one stored in the checkpoint.")
        device = next(self.backbone.parameters()).device
        self.classifier = self.classifier.to(device)

    def load_attention_pooling(self, checkpoint: dict):
        """Restore a saved AttentionPooling layer."""
        self.build_attention_pooling()
        missing, unexpected = self.attn_pool.load_state_dict(
            checkpoint["attention_pooling"], strict=False)
        if missing or unexpected:
            logger.warning(f"attention pooling restore skipped keys — missing "
                           f"{list(missing)}, unexpected {list(unexpected)}")
        device = next(self.backbone.parameters()).device
        self.attn_pool = self.attn_pool.to(device)
