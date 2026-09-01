"""
classifier.py

Configurable MLP classification head sitting on top of the VideoMAE backbone.
Maps the pooled backbone embedding to `num_classes` RAW LOGITS.

The head is deliberately task-agnostic: its width comes from
DataSpec.num_classes and it never applies an output activation. What changes
between tasks is what those logits MEAN, and that is expressed entirely through
the `bias` init and the loss:

    multiclass  1 + len(activities) logits, consumed by CrossEntropyLoss.
                bias = log(p_c)            — softmax log-prior, so an untrained
                                             head predicts the class frequencies.
    multilabel      len(activities) logits, consumed by BCEWithLogitsLoss.
                bias = log(p_c/(1-p_c))    — sigmoid logit-prior, so each
                                             independent activity starts at its
                                             own base rate. Critical here: the
                                             activities are rare, and a
                                             zero-bias sigmoid head starts at
                                             p=0.5 for every one of them.

Both bias vectors are computed from the TRAIN split by
VideoMAEDataset.compute_bias() and passed in by training.py.
"""

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    def __init__(self, in_dim, dims, num_classes, activation="relu", dropout=0.2, bias=None):
        """
        Args:
            in_dim      (int): input embedding dim (768 base / 1408 giant).
            dims        (list[int]): hidden layer widths; [] = single linear layer.
            num_classes (int): number of output logits (DataSpec.num_classes).
            activation  (str): HIDDEN-layer activation, "relu" | "gelu" | "tanh".
                               Not the output activation — see the module docstring.
            dropout     (float): dropout after each hidden activation.
            bias        (Tensor|None): optional (num_classes,) OUTPUT-bias init
                                       (log-prior or logit-prior, per task).
                                       None => the output layer is built with no
                                       bias term at all (`use_bias: false`).
                                       Hidden layers always keep theirs.
        """
        super().__init__()

        dims = [in_dim] + dims + [num_classes]
        act_lookup = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
        act_cls = act_lookup.get(activation.lower(), nn.ReLU)
        layers = []
        for i in range(len(dims) - 2):
            # Hidden layers ALWAYS keep their bias. `bias` is the output-layer
            # prior; gating every Linear on it silently stripped the biases from
            # the whole MLP whenever `use_bias: false` (or a checkpoint whose head
            # had none) was in play, which is a different architecture, not a
            # different initialisation.
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1], bias=(bias is not None)))
        if bias is not None:
            if bias.shape[-1] != num_classes:
                raise ValueError(
                    f"bias init has {bias.shape[-1]} entries but the head emits "
                    f"{num_classes} logits")
            with torch.no_grad():
                layers[-1].bias.copy_(bias)
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x (Tensor): (batch, in_dim) pooled clip embedding.
        Returns:
            Tensor: (batch, num_classes) RAW logits — no softmax/sigmoid applied.
                    Feed to CrossEntropyLoss / BCEWithLogitsLoss, or call
                    VideoModel.probs() for probabilities.
        """
        return self.seq(x)
