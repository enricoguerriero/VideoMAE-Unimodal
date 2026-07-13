"""
videomae.py

VideoMAE — wraps the MCG-NJU/videomae-base-finetuned-ssv2 encoder for
single-label 4-class activity classification (non_target / stimulation /
ventilation / suction) on 3-second clips.

Adapted from the multimodal repo: inherits the trimmed `VideoModel` base
(no LoRA), and `num_classes` now defaults to 4. The backbone/forward logic is
unchanged — it still emits raw logits; only the head width and the downstream
loss (CrossEntropy) differ.
"""

import torch
from transformers import VideoMAEModel, VideoMAEImageProcessor

from .base import VideoModel


class VideoMAE(VideoModel):
    def __init__(self, device: str = "cuda", num_classes: int = 4,
                 backbone_id: str = "MCG-NJU/videomae-base-finetuned-ssv2"):
        super().__init__(num_classes=num_classes, backbone_id=backbone_id, device=device)
        self.model_name = "VideoMAE"
        # Encoder only (no HF classification head) — we attach our own head.
        self.backbone = VideoMAEModel.from_pretrained(backbone_id, ignore_mismatched_sizes=True)
        self.processor = VideoMAEImageProcessor.from_pretrained(backbone_id)
        self.hidden_size = self.backbone.config.hidden_size  # 768 for base
        self.num_frames = 16  # fixed by architecture (8x196 position embeddings)
        self.input_device = torch.device(device if torch.cuda.is_available() else "cpu")

    def forward(self, pixel_values: torch.Tensor, **kwargs):
        """
        Args:
            pixel_values (Tensor): (B, 16, 3, 224, 224) as produced by the processor.
        Returns:
            Tensor: (B, num_classes) raw logits (feed to CrossEntropyLoss / argmax).
        """
        device = next(self.backbone.parameters()).device
        outputs = self.backbone(pixel_values=pixel_values.to(device), return_dict=True)
        cls_token = outputs.last_hidden_state[:, 0, :]  # (B, hidden_size)

        if self.attn_pool is not None:
            seq = outputs.last_hidden_state
            mask = torch.ones(seq.shape[:2], dtype=torch.bool, device=seq.device)
            pooled = self.attn_pool(seq, mask)
        else:
            pooled = cls_token

        logits = self.classifier(pooled.float())
        return logits

    def load_backbone(self, checkpoint: dict, config: dict = None):
        # No LoRA — load encoder weights directly.
        self.backbone.load_state_dict(checkpoint["backbone"], strict=False)
