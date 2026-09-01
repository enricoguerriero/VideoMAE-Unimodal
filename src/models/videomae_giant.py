"""
videomae_giant.py

VideoMAEGiant — wraps the OpenGVLab/VideoMAEv2-giant backbone (~1B params, ViT-g)
for neonatal resuscitation activity recognition.

Adapted from the multimodal repo: inherits the trimmed `VideoModel` base
(no LoRA). `num_classes` and `task` come from configs/data.yaml via DataSpec,
so the same backbone serves both the multiclass (softmax) and multilabel
(sigmoid) tasks. The three-step manual loading strategy
(bypassing AutoModel.from_pretrained's meta-device init, which is incompatible
with VideoMAEv2's custom __init__) is kept verbatim.

Extra deps: pip install timm easydict
"""

import logging

import torch
from transformers import VideoMAEImageProcessor, AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from .base import VideoModel

logger = logging.getLogger(__name__)


class VideoMAEGiant(VideoModel):
    def __init__(self, device: str = "cuda", num_classes: int = 4,
                 backbone_id: str = "OpenGVLab/VideoMAEv2-giant",
                 task: str = "multiclass"):
        super().__init__(num_classes=num_classes, backbone_id=backbone_id,
                         device=device, task=task)
        self.model_name = "VideoMAEGiant"

        # 1. Config
        config = AutoConfig.from_pretrained(backbone_id, trust_remote_code=True)

        # 2. Instantiate the model class directly on CPU (bypass meta device).
        model_class = get_class_from_dynamic_module("modeling_videomaev2.VideoMAEv2", backbone_id)
        self.backbone = model_class(config)

        # 3. Load weights manually from the cached safetensors file.
        weights_path = hf_hub_download(repo_id=backbone_id, filename="model.safetensors")
        state_dict = load_file(weights_path, device="cpu")
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        # UNEXPECTED keys are decoder weights (harmless for classification).
        # MISSING keys would signal a real mismatch.

        self.processor = VideoMAEImageProcessor.from_pretrained(backbone_id)
        self.hidden_size = config.model_config["embed_dim"]  # 1408 for ViT-g
        self.num_frames = 16
        self.input_device = torch.device(device if torch.cuda.is_available() else "cpu")

    def forward(self, pixel_values: torch.Tensor, **kwargs):
        """
        Args:
            pixel_values (Tensor): (B, 16, 3, 224, 224) in (B, T, C, H, W) order.
        Returns:
            Tensor: (B, num_classes) RAW logits — no output activation applied.
        """
        device = next(self.backbone.parameters()).device
        # VideoMAEv2 expects (B, C, T, H, W).
        pixel_values = pixel_values.to(device).permute(0, 2, 1, 3, 4)

        # Backbone returns a (B, hidden_size) mean-pooled feature vector — there is
        # no token sequence left to pool over, which is why attention pooling is
        # rejected outright (see build_attention_pooling).
        features = self.backbone(pixel_values=pixel_values)

        logits = self.classifier(features.float())
        return logits

    def build_attention_pooling(self):
        """Not available on this backbone — rejected instead of silently no-op'ing.

        VideoMAEv2-giant pools internally and hands back a single (B, hidden)
        vector. Wrapping AttentionPooling around it pools a length-1 sequence: the
        softmax over one element is the constant 1, so the module is an exact
        identity whose parameters get a gradient of exactly zero. It trained
        nothing and was written into every checkpoint as dead weight.
        """
        raise ValueError(
            "attention_pooling is not supported for VideoMAEGiant: the backbone "
            "already returns a pooled (B, hidden) vector, so pooling over it is an "
            "identity with zero gradient. Set `attention_pooling: false` in "
            "configs/config.yaml (and drop --attention_pooling).")

    def load_attention_pooling(self, checkpoint: dict):
        """Skip with a warning: any saved layer here was the identity described
        above, so restoring it would change nothing but would now raise."""
        logger.warning(
            "ignoring the saved attention-pooling layer — on VideoMAEGiant it was "
            "an identity over a length-1 sequence and never affected the output.")

    def load_backbone(self, checkpoint: dict, config: dict = None):
        self.backbone.load_state_dict(checkpoint["backbone"], strict=False)
