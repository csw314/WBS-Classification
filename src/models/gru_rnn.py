"""
src/models/gru_rnn.py
---------------------
Bidirectional GRU‑based sentence classifier for Work‑Breakdown‑Structure (WBS)
text.  Provides an alternative to the Kim‑style TextCNN so you can experiment
with *recurrent* architectures simply by changing ``Config.MODEL_ARCH``.

The class is intentionally plug‑and‑play with the existing training pipeline:
• Signature matches the factory call in ``src.train.build_model``.
• Extra **kwargs are accepted (and ignored) so `kernel_sizes`/`num_filters`
  passed by the CNN config do not raise an error.
• Weight initialisation mirrors best‑practice Xavier/KAIMING schemes.

Example
~~~~~~~
>>> from src.models.gru_rnn import GRURNN
>>> model = GRURNN(vocab_size=20_000, embed_dim=128, num_classes=8,
...                hidden_dim=256, num_layers=2, bidirectional=True)
>>> logits = model(torch.randint(0, 20_000, (32, 64)))
>>> logits.shape
torch.Size([32, 8])
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

__all__ = ["GRURNN", "GruRnn", "GRUClassifier"]


class GRURNN(nn.Module):
    """Bidirectional GRU network for sentence‑level classification.

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary.
    embed_dim : int
        Dimensionality of token embeddings.
    num_classes : int
        Number of output labels.
    hidden_dim : int, optional
        Hidden size of the GRU.  Default **256**.
    num_layers : int, optional
        Stack multiple recurrent layers.  Default **1**.
    bidirectional : bool, optional
        Use *bi‑directional* GRU so the model can see left & right context.
        Default **False**.
    dropout : float, optional
        Dropout probability applied **after** the recurrent stack.
        Default **0.5**.
    padding_idx : int, optional
        Index of the ``<pad>`` token so its embedding can be initialised to
        zeros and *excluded* from gradient updates.  Default **0**.
    **_: Any
        Placeholder to swallow extra keyword args (e.g. `kernel_sizes` passed
        by CNN config) to keep strict backwards compatibility with
        ``build_model``.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.5,
        padding_idx: int = 0,
        **_: Any,
    ) -> None:
        super().__init__()

        self.vocab_size   = vocab_size 
        self.num_classes  = num_classes

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx,
        )

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)
        fc_in_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(fc_in_dim, num_classes)

        self._init_weights()

    # ------------------------------------------------------------------ #
    # forward pass
    # ------------------------------------------------------------------ #
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Compute *unnormalised* class scores (logits).

        Parameters
        ----------
        input_ids : LongTensor of shape ``(batch, seq_len)``
            Batch of padded token‑id sequences.

        Returns
        -------
        logits : Tensor of shape ``(batch, num_classes)``
            Raw, *unnormalised* predictions suitable for ``F.cross_entropy``.
        """
        # Shape → (batch, seq_len, embed_dim)
        x = self.embedding(input_ids)

        # GRU expects (batch, seq_len, embed_dim) because batch_first=True
        output, hidden = self.gru(x)

        # hidden → (num_layers * num_directions, batch, hidden_dim)
        if self.gru.bidirectional:
            # Concatenate the final forward & backward hidden states
            h_fwd = hidden[-2]
            h_bwd = hidden[-1]
            h = torch.cat((h_fwd, h_bwd), dim=1)  # (batch, hidden_dim*2)
        else:
            h = hidden[-1]  # (batch, hidden_dim)

        logits = self.fc(self.dropout(h))
        return logits

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _init_weights(self) -> None:
        """Weight initialisation: uniform for embeddings, Xavier for linear, 
        Kaiming‑uniform for recurrent layers.
        """
        nn.init.uniform_(self.embedding.weight, -0.05, 0.05)

        def kaiming_rnn(param: torch.Tensor) -> None:
            if param.data.dim() >= 2:
                nn.init.kaiming_uniform_(param.data)
            else:
                nn.init.zeros_(param.data)

        for name, param in self.gru.named_parameters():
            if "weight" in name:
                kaiming_rnn(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)


# ---------------------------------------------------------------------- #
# Backwards‑compat aliases so legacy imports do not break
# ---------------------------------------------------------------------- #
GruRnn = GRURNN  # pylint: disable=invalid-name
GRUClassifier = GRURNN
