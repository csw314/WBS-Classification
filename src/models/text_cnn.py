"""
src/text_cnn.py – Convolutional Neural Network baseline for WBS-Classifier
-------------------------------------------------------------------------
Minor bug-fix (2025-07-29)
* **Bias initialisation guard** – ``nn.init.constant_(conv.bias, 0.0)`` is now
  executed *only* when ``conv.bias is not None`` so the model works even when
  `bias=False` is passed in future experiments.
* Padding embedding row is zero-initialised.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TextCNN"]


class TextCNN(nn.Module):
    """A simple yet strong TextCNN classifier.

    Parameters
    ----------
    vocab_size: int
        Size of the vocabulary.
    embed_dim: int
        Dimension of word embeddings.
    num_classes: int
        Number of output classes.
    kernel_sizes: tuple[int, ...], default=(3, 4, 5)
        Heights of convolution kernels (width is implicitly *embed_dim*).
    num_filters: int, default=100
        Number of filters per kernel size.
    dropout: float, default=0.5
        Dropout probability applied after pooling.
    padding_idx: int, default=0
        Index of the ``<pad>`` token – its embedding vector is initialised to
        zeros.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        *,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        num_filters: int = 100,
        dropout: float = 0.5,
        padding_idx: int = 0,
        **unused,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, embed_dim), bias=True) for k in kernel_sizes]
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

        self._init_weights()

    # ------------------------------------------------------------------ #
    # Weight initialisation                                              #
    # ------------------------------------------------------------------ #

    def _init_weights(self) -> None:
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        # Zero-initialise PAD row for stability
        with torch.no_grad():
            self.embedding.weight[self.embedding.padding_idx].zero_()

        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
            if conv.bias is not None:  # ← guard added
                nn.init.constant_(conv.bias, 0.0)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    # ------------------------------------------------------------------ #
    # Forward pass                                                      #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # shape: (batch, seq_len)
        embedded = self.embedding(x).unsqueeze(1)  # (batch, 1, seq_len, embed_dim)

        conv_outs = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(co, co.size(2)).squeeze(2) for co in conv_outs]
        cat = torch.cat(pooled, dim=1)
        dropped = self.dropout(cat)
        return self.fc(dropped)

