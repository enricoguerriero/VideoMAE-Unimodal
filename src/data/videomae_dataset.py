"""
videomae_dataset.py

PyTorch Dataset feeding 3-second MP4 clips into the VideoMAE backbones for
SINGLE-LABEL 4-class activity classification (non_target / stimulation /
ventilation / suction).

Adapted from the multimodal repo's VideoMAEDataset:
    * The CSV manifest has columns `video_path,label` where `label` is a single
      integer class index in {0,1,2,3} (NOT three binary columns).
    * __getitem__ returns "labels" as a scalar int64 tensor (a class index),
      consumed by CrossEntropyLoss / argmax — not a (3,) multi-label vector.
    * compute_class_weights() returns inverse-frequency weights for the
      CrossEntropyLoss `weight` argument; compute_bias() returns per-class
      log-priors for the softmax head bias init.

VideoMAE requires exactly 16 frames per clip; frame indices are pre-computed
with np.linspace over each clip's frame count for uniform temporal coverage.
"""

import av
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class VideoMAEDataset(Dataset):

    NUM_CLASSES = 4

    def __init__(self, video_csv: str, processor, num_frames: int = 16):
        """
        Args:
            video_csv (str): CSV manifest with columns `video_path`, `label`.
            processor: HuggingFace VideoMAEImageProcessor.
            num_frames (int): ignored — hard-fixed to 16 (VideoMAE requirement).
        """
        super().__init__()
        self.processor = processor
        self.data = pd.read_csv(video_csv)
        self.labels = self._build_labels(self.data)
        self.num_frames = 16  # VideoMAE requires exactly 16 frames
        self.videos, self.indices = self._prepare_videos(
            self.data["video_path"].tolist(), self.num_frames
        )

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------
    @staticmethod
    def _build_labels(df):
        """Return (N,) int64 tensor of class indices."""
        return torch.tensor(df["label"].astype(int).values, dtype=torch.long)

    @classmethod
    def _get_label_counts(cls, df):
        """Return (counts (C,) float32, n int) — positive count per class."""
        labels = df["label"].astype(int).values
        counts = np.bincount(labels, minlength=cls.NUM_CLASSES).astype(np.float32)
        return torch.tensor(counts, dtype=torch.float32), len(df)

    def compute_class_weights(self):
        """
        sqrt inverse-frequency class weights for CrossEntropyLoss(weight=...),
        matching the multimodal thesis's MoViNet video base model exactly:
            weight_c = sqrt(n_total / (num_classes * count_c))
        (no post-normalisation). Classes absent from the split get weight 0.
        """
        counts, n = self._get_label_counts(self.data)
        weights = torch.where(
            counts > 0,
            torch.sqrt(n / (self.NUM_CLASSES * counts.clamp(min=1e-6))),
            torch.zeros_like(counts),
        )
        return weights.float()

    def compute_bias(self):
        """Per-class log-prior log(count_c / n) for softmax head bias init."""
        counts, n = self._get_label_counts(self.data)
        priors = counts.clamp(min=1e-6) / max(n, 1)
        return torch.log(priors).float()

    # ------------------------------------------------------------------
    # Frames
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.data)

    @staticmethod
    def _prepare_videos(paths, num_frames):
        """Pre-compute uniformly spaced frame indices for each clip."""
        idxs, out_paths = [], []
        for p in paths:
            container = av.open(p)
            total = container.streams.video[0].frames
            container.close()
            if total is None or total <= 0:
                # Some re-encoded (mp4v) clips report 0 frames in the header;
                # fall back to a full decode-count so linspace stays valid.
                total = _count_frames_by_decode(p)
            indices = np.linspace(0, max(total - 1, 0), num_frames, dtype=int)
            idxs.append(indices)
            out_paths.append(p)
        return out_paths, idxs

    def _read_frames_at_indices(self, filepath, indices):
        """Decode only the requested frames; pad with the last frame if short."""
        container = av.open(filepath)
        frames = []
        target = set(int(i) for i in indices)
        last_idx = int(indices[-1])
        for i, frm in enumerate(container.decode(video=0)):
            if i > last_idx:
                break
            if i in target:
                frames.append(frm.to_ndarray(format="rgb24"))
        container.close()
        if len(frames) == 0:
            raise RuntimeError(f"Decoded 0 frames from {filepath}")
        if len(frames) < self.num_frames:
            frames += [frames[-1]] * (self.num_frames - len(frames))
        return np.stack(frames[: self.num_frames])

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        video_path = self.videos[idx]
        frame_indices = self.indices[idx]
        frames = self._read_frames_at_indices(video_path, frame_indices)
        inputs = self.processor(list(frames), return_tensors="pt")
        return {
            "pixel_values": inputs.pixel_values.squeeze(0),  # (16, 3, 224, 224)
            "labels": self.labels[idx],                      # scalar int64
        }


def _count_frames_by_decode(path: str) -> int:
    """Count frames by decoding (robust for mp4v clips with a 0-frame header)."""
    container = av.open(path)
    n = sum(1 for _ in container.decode(video=0))
    container.close()
    return n
