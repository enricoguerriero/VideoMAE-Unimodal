"""
collate.py

Custom collate for the VideoMAE DataLoader. Simpler than the multimodal repo's
version because there is only one model family (pure ViT): each sample carries
`pixel_values`, `labels`, and — in multilabel mode — `label_mask`.

Shapes out, by task (set by configs/data.yaml):
    multiclass  labels (B,)   int64    class indices
    multilabel  labels (B, C) float32  independent 0/1 targets
                label_mask (B, C) float32  1 = supervise, 0 = exclude from loss

The task is never inspected here — whatever the Dataset put in each item is
stacked, and `label_mask` is simply absent in multiclass.
"""

import torch


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack a list of Dataset items into a batch dict."""
    collated = {
        "labels": torch.stack([item["labels"] for item in batch]),
    }
    if "pixel_values" in batch[0]:
        collated["pixel_values"] = torch.stack([item["pixel_values"] for item in batch])
    if "label_mask" in batch[0]:
        collated["label_mask"] = torch.stack([item["label_mask"] for item in batch])
    return collated
