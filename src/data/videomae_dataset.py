"""
videomae_dataset.py

PyTorch Dataset feeding 3-second MP4 clips into the VideoMAE backbones, for
EITHER task defined by configs/data.yaml:

    multiclass : "labels" is a scalar int64 class index in [0, num_classes)
                 -> softmax head + weighted CrossEntropyLoss + argmax
    multilabel : "labels" is a (num_activities,) float32 vector of 0/1 and
                 "label_mask" is a (num_activities,) float32 vector where 1
                 means "supervise this activity for this clip" and 0 means
                 "exclude it from the loss"
                 -> sigmoid head + masked BCEWithLogitsLoss + per-class thresholds

Labels are NOT read from the manifest. The manifest carries EVIDENCE — each
clip's original `bucket` and its per-activity `frac_*` window coverage — and the
DataSpec turns that into targets HERE, at load time. Changing task, thresholds,
the ambiguous-band policy or the bucket keep/drop list therefore needs no
manifest rebuild: `self.data` is filtered on construction and clips the current
config rejects simply never appear in the epoch.

A legacy manifest with a plain `label` column (pre-DataSpec) is still accepted in
multiclass mode; see `_resolve_rows`.

VideoMAE requires exactly 16 frames per clip; frame indices are pre-computed
with np.linspace over each clip's frame count for uniform temporal coverage.
"""

import logging

import av
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .spec import DataSpec

logger = logging.getLogger(__name__)

WEIGHTINGS = ("sqrt_inv_freq", "inv_freq", "none")


class VideoMAEDataset(Dataset):

    def __init__(self, video_csv: str, processor, spec: DataSpec, num_frames: int = 16):
        """
        Args:
            video_csv (str | pd.DataFrame): manifest CSV path, or an already
                loaded/filtered DataFrame. Task-agnostic columns
                `video_path,bucket,frac_<activity>...` (build_manifest.py), or a
                legacy `video_path,label` CSV in multiclass mode.
            processor: HuggingFace VideoMAEImageProcessor.
            spec (DataSpec): loaded configs/data.yaml — decides the targets.
            num_frames (int): ignored — hard-fixed to 16 (VideoMAE requirement).
        """
        super().__init__()
        self.processor = processor
        self.spec = spec
        self.num_frames = 16  # VideoMAE requires exactly 16 frames

        # A DataFrame is accepted so callers can evaluate a SUBSET of a manifest
        # (e.g. `df[df.thesis_test == 1]`) without writing a temporary CSV.
        raw = video_csv if isinstance(video_csv, pd.DataFrame) else pd.read_csv(video_csv)
        source = "<dataframe>" if isinstance(video_csv, pd.DataFrame) else video_csv
        self.data, self.labels, self.masks, self.n_dropped = self._resolve_rows(raw, spec)
        if len(self.data) == 0:
            raise ValueError(
                f"{source}: no clips survived the data config ({spec.source}). "
                f"Every one of its {len(raw)} rows was dropped — check `buckets` "
                f"and `thresholds` in the data config.")
        if self.n_dropped:
            logger.info(f"{source}: {len(self.data)} clips kept, "
                        f"{self.n_dropped} dropped by {spec.source}")

        self.videos, self.indices = self._prepare_videos(
            self.data["video_path"].tolist(), self.num_frames
        )

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_rows(raw: pd.DataFrame, spec: DataSpec):
        """Apply the DataSpec row by row.

        Returns (kept_df, labels, masks, n_dropped) where `labels` is (N,) int64
        in multiclass and (N, C) float32 in multilabel, and `masks` is (N, C)
        float32 in multilabel / None in multiclass.
        """
        frac_cols = spec.frac_columns()
        has_evidence = "bucket" in raw.columns and all(c in raw.columns for c in frac_cols)

        if not has_evidence:
            if "label" not in raw.columns:
                raise ValueError(
                    f"manifest has neither the evidence columns "
                    f"({['bucket'] + frac_cols}) nor a legacy `label` column. "
                    f"Rebuild it with `python -m src.data.build_manifest ...`.")
            if spec.is_multilabel:
                raise ValueError(
                    "this manifest is a legacy single-label CSV (`label` column only) "
                    "and carries no per-activity fractions, so multilabel targets "
                    "cannot be derived. Rebuild it with "
                    "`python -m src.data.build_manifest ...`.")
            logger.warning(
                "legacy manifest: using its `label` column verbatim. `buckets`, "
                "`thresholds` and `ambiguous` in the data config are IGNORED.")
            labels = torch.tensor(raw["label"].astype(int).values, dtype=torch.long)
            bad = labels[(labels < 0) | (labels >= spec.num_classes)]
            if len(bad):
                raise ValueError(
                    f"legacy manifest has label values outside "
                    f"[0, {spec.num_classes}): e.g. {bad[:5].tolist()}")
            return raw.reset_index(drop=True), labels, None, 0

        # `tagged`/`clip_dir` are absent from manifests built before fraction tags
        # were handled per site; assume tagged (the old behaviour) when missing.
        has_tag_cols = "tagged" in raw.columns and "clip_dir" in raw.columns
        keep_idx, targets, masks = [], [], []
        for i, row in enumerate(raw.itertuples(index=False)):
            fracs = {a: float(getattr(row, f"frac_{a}")) for a in spec.activities}
            tagged = bool(int(getattr(row, "tagged"))) if has_tag_cols else True
            dir_acts = (spec.activities_from_path(getattr(row, "clip_dir"))
                        if has_tag_cols else ())
            label = spec.resolve(int(row.bucket), fracs, tagged=tagged,
                                 dir_activities=dir_acts)
            if label is None:
                continue
            keep_idx.append(i)
            if spec.is_multilabel:
                targets.append(label.targets)
                masks.append(label.mask)
            else:
                targets.append(label.class_index)

        kept = raw.iloc[keep_idx].reset_index(drop=True)
        n_dropped = len(raw) - len(kept)
        if spec.is_multilabel:
            return (kept,
                    torch.tensor(targets, dtype=torch.float32),
                    torch.tensor(masks, dtype=torch.float32),
                    n_dropped)
        return kept, torch.tensor(targets, dtype=torch.long), None, n_dropped

    # ------------------------------------------------------------------
    # Class statistics (loss weights + head bias init)
    # ------------------------------------------------------------------
    def label_counts(self):
        """Positive count per output unit, as a (num_classes,) float tensor."""
        if self.spec.is_multilabel:
            return (self.labels * self.masks).sum(dim=0)
        return torch.bincount(self.labels, minlength=self.spec.num_classes).float()

    def supervised_counts(self):
        """Supervised (unmasked) example count per output unit.

        multiclass: the split size, repeated — every clip supervises every logit
        through the softmax. multilabel: per-activity, masked clips excluded.
        """
        if self.spec.is_multilabel:
            return self.masks.sum(dim=0)
        return torch.full((self.spec.num_classes,), float(len(self.labels)))

    def compute_class_weights(self, weighting: str = "sqrt_inv_freq"):
        """CrossEntropyLoss `weight` (multiclass only).

        sqrt inverse-frequency, matching the multimodal thesis's MoViNet video
        base model exactly:  weight_c = sqrt(n_total / (num_classes * count_c))
        (no post-normalisation). Classes absent from the split get weight 0.
        """
        if self.spec.is_multilabel:
            raise RuntimeError("compute_class_weights() is multiclass-only; "
                               "use compute_pos_weight() for multilabel")
        if weighting not in WEIGHTINGS:
            raise ValueError(f"weighting must be one of {WEIGHTINGS}, got {weighting!r}")
        counts = self.label_counts()
        n, c = float(len(self.labels)), self.spec.num_classes
        if weighting == "none":
            return torch.ones(c, dtype=torch.float32)
        ratio = n / (c * counts.clamp(min=1e-6))
        weights = torch.sqrt(ratio) if weighting == "sqrt_inv_freq" else ratio
        return torch.where(counts > 0, weights, torch.zeros_like(weights)).float()

    def compute_pos_weight(self, weighting: str = "sqrt_inv_freq"):
        """BCEWithLogitsLoss `pos_weight` (multilabel only): per activity,
        neg/pos over the SUPERVISED examples, sqrt-damped by default to mirror
        the thesis' sqrt inverse-frequency class weighting.
        """
        if not self.spec.is_multilabel:
            raise RuntimeError("compute_pos_weight() is multilabel-only; "
                               "use compute_class_weights() for multiclass")
        if weighting not in WEIGHTINGS:
            raise ValueError(f"weighting must be one of {WEIGHTINGS}, got {weighting!r}")
        pos = self.label_counts()
        neg = (self.supervised_counts() - pos).clamp(min=0.0)
        if weighting == "none":
            return torch.ones(self.spec.num_classes, dtype=torch.float32)
        ratio = neg / pos.clamp(min=1e-6)
        weights = torch.sqrt(ratio) if weighting == "sqrt_inv_freq" else ratio
        return torch.where(pos > 0, weights, torch.ones_like(weights)).float()

    def compute_bias(self):
        """Output-layer bias init that anchors the head at the empirical priors.

        multiclass: log(p_c)                  — softmax log-prior
        multilabel: log(p_c / (1 - p_c))      — sigmoid logit-prior
        """
        pos = self.label_counts()
        if self.spec.is_multilabel:
            total = self.supervised_counts().clamp(min=1.0)
            p = (pos / total).clamp(min=1e-6, max=1 - 1e-6)
            return torch.log(p / (1 - p)).float()
        p = (pos / max(len(self.labels), 1)).clamp(min=1e-6)
        return torch.log(p).float()

    def describe_labels(self) -> str:
        """One-line-per-unit summary of the resolved targets, for the log."""
        pos, sup = self.label_counts(), self.supervised_counts()
        lines = []
        for i, name in enumerate(self.spec.class_names):
            frac = float(pos[i]) / max(float(sup[i]), 1.0)
            extra = ""
            if self.spec.is_multilabel:
                masked = len(self.labels) - int(sup[i])
                extra = f", {masked} masked out"
            lines.append(f"    {name:<16} {int(pos[i]):>8,} positives "
                         f"({100 * frac:5.2f}%{extra})")
        if self.spec.is_multilabel:
            n_active = (self.labels * self.masks).sum(dim=1)
            for k in range(self.spec.num_classes + 1):
                n = int((n_active == k).sum())
                if n:
                    lines.append(f"    clips with {k} active activit"
                                 f"{'y' if k == 1 else 'ies'}: {n:,}")
        return "\n".join(lines)

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
        item = {
            "pixel_values": inputs.pixel_values.squeeze(0),  # (16, 3, 224, 224)
            "labels": self.labels[idx],                      # scalar int64 | (C,) float32
        }
        if self.masks is not None:
            item["label_mask"] = self.masks[idx]             # (C,) float32
        return item


def _count_frames_by_decode(path: str) -> int:
    """Count frames by decoding (robust for mp4v clips with a 0-frame header)."""
    container = av.open(path)
    n = sum(1 for _ in container.decode(video=0))
    container.close()
    return n
