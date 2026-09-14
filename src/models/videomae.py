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

LEGACY POOLING: checkpoints trained before that fix (before 2026-08-31) fitted
their head on `last_hidden_state[:, 0]` — the FIRST PATCH TOKEN, read as if it
were a CLS token. That feature space is unrelated to the mean-pooled one, so
scoring such a checkpoint under the current pooling produces garbage (typically
a collapse onto the two most frequent classes). `pooling="patch0_legacy"`
reproduces the old path so those checkpoints remain evaluable; the default
`"auto"` detects them by the absence of `fc_norm` in the saved backbone and
switches for you, loudly. It is a compatibility shim, not a training option —
new runs must use the mean pooling the backbone was pretrained with.
"""

import logging

import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from .base import VideoModel

logger = logging.getLogger(__name__)

# Pooling modes. "auto" is a POLICY resolved at checkpoint-restore time; the two
# below it are the actual feature paths `forward()` can take.
POOLING_AUTO = "auto"
POOLING_MEAN = "mean"            # mean over all patch tokens -> fc_norm (correct)
POOLING_PATCH0 = "patch0_legacy"  # last_hidden_state[:, 0] (pre-2026-08-31 runs)
POOLING_MODES = (POOLING_AUTO, POOLING_MEAN, POOLING_PATCH0)


class VideoMAE(VideoModel):
    def __init__(self, device: str = "cuda", num_classes: int = 4,
                 backbone_id: str = "MCG-NJU/videomae-base-finetuned-ssv2",
                 task: str = "multiclass", pooling: str = POOLING_AUTO):
        super().__init__(num_classes=num_classes, backbone_id=backbone_id,
                         device=device, task=task)
        self.model_name = "VideoMAE"
        if pooling not in POOLING_MODES:
            raise ValueError(f"pooling must be one of {POOLING_MODES}, got {pooling!r}")
        # The policy is remembered so `load_backbone` knows whether it is allowed
        # to switch; `self.pooling` is the mode forward() actually uses and starts
        # at the correct one, so a from-scratch run is never in legacy mode.
        self._pooling_policy = pooling
        self.pooling = POOLING_PATCH0 if pooling == POOLING_PATCH0 else POOLING_MEAN
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
        device = self.forward_device(pixel_values)
        outputs = self.backbone(pixel_values=pixel_values.to(device), return_dict=True)
        seq = outputs.last_hidden_state  # (B, 1568, hidden_size) — all patch tokens

        if self.attn_pool is not None:
            mask = torch.ones(seq.shape[:2], dtype=torch.bool, device=seq.device)
            pooled = self.attn_pool(seq, mask)
        elif self.pooling == POOLING_PATCH0:
            # COMPATIBILITY ONLY — reproduces the pre-fix feature so a head
            # trained against it stays meaningful. See the module docstring.
            pooled = seq[:, 0, :]
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

        # A checkpoint written before the pooling fix has no `fc_norm` in its
        # backbone, because it was saved from a bare VideoMAEModel. That absence
        # is a reliable tell: the head inside it was fitted on seq[:, 0].
        # Attention-pooled runs are exempt — their pooling is a saved module, so
        # which token path the trunk would otherwise use is irrelevant. Read the
        # flag off `config` rather than `self.attn_pool`, which test.py only
        # populates after this call.
        attn_pooled = bool((config or {}).get("attention_pooling", False))
        pre_fix = (getattr(self.backbone, "fc_norm", None) is not None
                   and not any(k.startswith("fc_norm.") for k in saved))

        if pre_fix and not attn_pooled and self._pooling_policy == POOLING_AUTO:
            self.pooling = POOLING_PATCH0
            logger.warning(
                "this checkpoint has no `fc_norm` in its backbone, so it was trained "
                "before the 2026-08-31 pooling fix — its head was fitted on seq[:, 0] "
                "(one patch token), not on the mean over all 1568 tokens. Switching "
                "this run to legacy `patch0` pooling so the head is fed the feature it "
                "was trained on. These numbers are comparable with that run's own "
                "history, NOT with post-fix checkpoints; retrain to get the correct "
                "pooling. Pass --legacy-pooling off to score it under mean pooling "
                "anyway (predictions will be meaningless).")
        elif pre_fix and not attn_pooled and self.pooling == POOLING_MEAN:
            logger.warning(
                "this checkpoint has no `fc_norm` in its backbone, so it was trained "
                "before the pooling fix — its head was fitted on seq[:, 0] (one patch "
                "token) and is now being fed the mean over all 1568 tokens. The two "
                "feature spaces are unrelated; expect meaningless predictions. Pass "
                "--legacy-pooling on to score it the way it was trained, or retrain.")
        elif self.pooling == POOLING_PATCH0 and not attn_pooled:
            logger.warning(
                "legacy `patch0` pooling is FORCED for this run: the clip feature is "
                "last_hidden_state[:, 0], not the mean over patch tokens. Only correct "
                "for checkpoints trained before 2026-08-31.")

        missing, unexpected = self.backbone.load_state_dict(saved, strict=False)
        if missing or unexpected:
            logger.warning(f"backbone restore skipped keys — missing {list(missing)[:8]}"
                           f"{'...' if len(missing) > 8 else ''}, unexpected "
                           f"{list(unexpected)[:8]}{'...' if len(unexpected) > 8 else ''}")
