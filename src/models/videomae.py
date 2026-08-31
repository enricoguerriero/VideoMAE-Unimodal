"""
videomae.py

VideoMAE — wraps the MCG-NJU/videomae-base-finetuned-ssv2 encoder for neonatal
resuscitation activity recognition on 3-second clips.

Adapted from the multimodal repo: inherits the trimmed `VideoModel` base
(no LoRA). `num_classes` and `task` come from configs/data.yaml via DataSpec
(see src/utils/model_loading.py), so the same backbone serves both the
multiclass (softmax, 1+N logits) and multilabel (sigmoid, N logits) tasks.
The backbone emits raw logits; only the head width and the downstream loss
differ between tasks.

POOLING: VideoMAE has no CLS token. Clip features are the MEAN over all 1568
patch tokens followed by the pretrained `fc_norm` — the same path
`VideoMAEForVideoClassification` uses, so the encoder is consumed the way it was
trained. Set `attention_pooling: true` to learn a pooling over the same tokens
instead.
"""

import logging

import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from .base import VideoModel

logger = logging.getLogger(__name__)


class VideoMAE(VideoModel):
    def __init__(self, device: str = "cuda", num_classes: int = 4,
                 backbone_id: str = "MCG-NJU/videomae-base-finetuned-ssv2",
                 task: str = "multiclass"):
        super().__init__(num_classes=num_classes, backbone_id=backbone_id,
                         device=device, task=task)
        self.model_name = "VideoMAE"
        # Load the CLASSIFICATION wrapper, then keep its encoder and its fc_norm.
        #
        # VideoMAE has NO CLS token — `last_hidden_state` is 1568 patch tokens
        # (8 temporal x 196 spatial) and nothing else. The pretrained model pools
        # by MEAN over those tokens followed by `fc_norm`, and that LayerNorm
        # lives on the classification wrapper, not on VideoMAEModel. Loading the
        # bare VideoMAEModel therefore silently discards it (it shows up as
        # `fc_norm.weight | UNEXPECTED` in the transformers load report) and
        # leaves no correct way to reproduce the pretraining pooling.
        #
        # fc_norm is attached to `self.backbone` rather than held separately so
        # the rest of the repo needs no changes: it is then covered by
        # `backbone.state_dict()` when checkpointing, by `backbone.parameters()`
        # in the backbone LR group, and by `load_backbone`'s strict=False restore.
        _full = VideoMAEForVideoClassification.from_pretrained(
            backbone_id, ignore_mismatched_sizes=True)
        self.backbone = _full.videomae
        self.backbone.fc_norm = _full.fc_norm  # LayerNorm, or None if not mean-pooled
        del _full
        self.processor = VideoMAEImageProcessor.from_pretrained(backbone_id)
        self.hidden_size = self.backbone.config.hidden_size  # 768 for base
        self.num_frames = 16  # fixed by architecture (8x196 position embeddings)
        self.input_device = torch.device(device if torch.cuda.is_available() else "cpu")

    def forward(self, pixel_values: torch.Tensor, **kwargs):
        """
        Args:
            pixel_values (Tensor): (B, 16, 3, 224, 224) as produced by the processor.
        Returns:
            Tensor: (B, num_classes) RAW logits — no output activation applied.
                    Feed to the task's loss, or call `self.probs()` for
                    softmax/sigmoid probabilities.
        """
        device = next(self.backbone.parameters()).device
        outputs = self.backbone(pixel_values=pixel_values.to(device), return_dict=True)
        seq = outputs.last_hidden_state  # (B, 1568, hidden_size) — all patch tokens

        if self.attn_pool is not None:
            mask = torch.ones(seq.shape[:2], dtype=torch.bool, device=seq.device)
            pooled = self.attn_pool(seq, mask)
        else:
            # The pretraining pooling: mean over every patch token, then fc_norm.
            # Taking seq[:, 0] instead would read one patch as if it were a CLS
            # token — VideoMAE has none, and the weights were never trained that way.
            pooled = seq.mean(dim=1)
            if getattr(self.backbone, "fc_norm", None) is not None:
                pooled = self.backbone.fc_norm(pooled)

        logits = self.classifier(pooled.float())
        return logits

    def load_backbone(self, checkpoint: dict, config: dict = None):
        # No LoRA — load encoder weights directly.
        saved = checkpoint["backbone"]
        if getattr(self.backbone, "fc_norm", None) is not None and \
                not any(k.startswith("fc_norm.") for k in saved):
            logger.warning(
                "this checkpoint has no `fc_norm` in its backbone, so it was trained "
                "before the pooling fix — its head was fitted on seq[:, 0] (one patch "
                "token) and is now being fed the mean over all 1568 tokens. The two "
                "feature spaces are unrelated; expect meaningless predictions. "
                "Retrain rather than evaluating this checkpoint.")
        self.backbone.load_state_dict(saved, strict=False)
