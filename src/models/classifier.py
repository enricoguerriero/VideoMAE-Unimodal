"""
classifier.py

Configurable MLP classification head sitting on top of the VideoMAE backbone.
Maps the pooled backbone embedding to raw logits for the FOUR mutually-exclusive
activity classes: non_target (0), stimulation (1), ventilation (2), suction (3).

Difference vs. the multimodal repo:
    * The head now emits `num_classes` (=4) logits for a single-label softmax
      task instead of 3 independent sigmoid logits. The layer stack itself is
      unchanged; only the meaning of the output layer differs (consumed by
      CrossEntropyLoss instead of BCEWithLogitsLoss).
    * The optional `bias` init is now the per-class log-prior (log p_c), which
      anchors the softmax at the empirical class frequencies.
"""

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    def __init__(self, in_dim, dims, num_classes, activation="relu", dropout=0.2, bias=None):
        """
        Args:
            in_dim      (int): input embedding dim (768 base / 1408 giant).
            dims        (list[int]): hidden layer widths; [] = single linear layer.
            num_classes (int): number of output logits (4).
            activation  (str): "relu" | "gelu" | "tanh" (default relu).
            dropout     (float): dropout after each hidden activation.
            bias        (Tensor|None): optional (num_classes,) output-bias init
                                       (log-priors). None => default zero bias.
        """
        super().__init__()

        dims = [in_dim] + dims + [num_classes]
        act_lookup = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
        act_cls = act_lookup.get(activation.lower(), nn.ReLU)
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=(bias is not None)))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1], bias=(bias is not None)))
        if bias is not None:
            with torch.no_grad():
                layers[-1].bias.copy_(bias)
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x (Tensor): (batch, in_dim) pooled clip embedding.
        Returns:
            Tensor: (batch, num_classes) raw logits for CrossEntropyLoss / argmax.
        """
        return self.seq(x)
