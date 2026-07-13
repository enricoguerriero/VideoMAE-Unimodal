"""
base.py

VideoModel — abstract base class for the VideoMAE backbones in this repo.

This is a trimmed, self-contained version of the multimodal repo's
`VisionLanguageModel`: all VLM/LoRA/PEFT machinery has been removed because this
project only fine-tunes pure vision transformers (VideoMAE-base, VideoMAEv2-giant)
with no language stream. Removing PEFT also removes a heavy dependency, keeping
the repo standalone.

Shared responsibilities kept here:
    * classifier-head construction (build_classifier)
    * optional attention pooling (build_attention_pooling / pooling)
    * checkpoint restore helpers (load_backbone / load_classifier /
      load_attention_pooling)

Concrete subclasses (videomae.py, videomae_giant.py) implement only their
backbone-specific __init__ and forward.
"""

import torch
import torch.nn as nn

from .classifier import ClassifierHead
from .attentionpooling import AttentionPooling


class VideoModel(nn.Module):
    def __init__(self, num_classes: int = 4, backbone_id: str = None, device=None):
        """
        Args:
            num_classes (int): number of output logits (4: non_target,
                               stimulation, ventilation, suction).
            backbone_id (str|None): HF model id, stored for reference.
            device (str|None): device string, stored for reference; concrete
                               placement handled by each subclass __init__.
        """
        super().__init__()
        self.device = device
        self.model_name = "VideoModel"
        self.num_classes = num_classes
        self.backbone_id = backbone_id
        self.backbone = None
        self.processor = None
        self.hidden_size = None
        self.classifier = None
        self.attn_pool = None
        self.input_device = None

    # ------------------------------------------------------------------
    # Pooling
    # ------------------------------------------------------------------
    def pooling(self, x, mask):
        """Masked mean pool, or learned attention pool if built."""
        if self.attn_pool is not None:
            return self.attn_pool(x, mask)
        pooled = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return pooled

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
        """Restore backbone weights (no LoRA). Subclasses may override."""
        self.backbone.load_state_dict(checkpoint["backbone"], strict=False)

    def load_classifier(self, checkpoint: dict, config: dict = None):
        """Build the head from config, restore its weights, move to backbone device."""
        classifier_config = (config or {}).get("classifier_config", {})
        self.build_classifier(classifier_config)
        self.classifier.load_state_dict(checkpoint["classifier"], strict=False)
        device = next(self.backbone.parameters()).device
        self.classifier = self.classifier.to(device)

    def load_attention_pooling(self, checkpoint: dict):
        """Restore a saved AttentionPooling layer."""
        self.build_attention_pooling()
        self.attn_pool.load_state_dict(checkpoint["attention_pooling"], strict=False)
        device = next(self.backbone.parameters()).device
        self.attn_pool = self.attn_pool.to(device)
