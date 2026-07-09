"""
collate.py

Custom collate for the VideoMAE DataLoader. Simpler than the multimodal repo's
version because there is only one model family (pure ViT): each sample carries
`pixel_values` and a scalar class `labels`.

`labels` is stacked into a (B,) int64 tensor (single-label class indices),
NOT a (B, 3) float multi-label tensor.
"""

import torch


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Args:
        batch: list of dicts each with "pixel_values" (16,3,224,224) and
               "labels" (scalar int64 class index in {0,1,2,3}).
    Returns:
        dict with "pixel_values" (B,16,3,224,224) and "labels" (B,) int64.
    """
    collated = {
        "labels": torch.stack([item["labels"] for item in batch]),
    }
    if "pixel_values" in batch[0]:
        collated["pixel_values"] = torch.stack([item["pixel_values"] for item in batch])
    return collated
