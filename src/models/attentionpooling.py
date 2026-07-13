"""
attentionpooling.py

Learnable attention pooling that collapses a variable-length sequence of token
embeddings into a single fixed-size vector. Used as an optional temporal
aggregation head for the VideoMAE backbones, replacing the default CLS/mean
pooling with a learned weighted combination over frames/patches.

Unchanged from the multimodal-repo implementation — it is task-agnostic and works
identically for single-label 4-class classification.
"""

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        """
        Args:
            hidden_dim (int): dimensionality of the input token embeddings
                              (768 for VideoMAE-base, 1408 for VideoMAEv2-giant).
        """
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask=None):
        """
        Args:
            x    (Tensor): (batch, seq_len, hidden_dim) token embeddings.
            mask (BoolTensor|None): (batch, seq_len); True = valid, False = padded.
        Returns:
            Tensor: (batch, hidden_dim) float32 pooled vector.
        """
        attn_weights = self.attn(x).squeeze(-1)  # (batch, seq_len)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask, float("-inf"))
        attn_scores = torch.softmax(attn_weights, dim=1)  # (batch, seq_len)
        pooled = (x * attn_scores.unsqueeze(-1)).sum(1)   # (batch, hidden_dim)
        return pooled.float()
