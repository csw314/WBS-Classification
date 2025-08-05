"""
src/models/lstm_rnn.py
----------------------
Bidirectional **LSTM** sentence-classifier compatible with the existing
training pipeline (see `src/train.py`).  Mirrors the API of `GRURNN` so you
can swap architectures simply by setting `Config.MODEL_ARCH = "lstm_rnn"`.

Why LSTM?
~~~~~~~~~
Long-Short-Term-Memory cells keep separate *cell* and *hidden* states which
helps retain long-range information compared with vanilla RNNs/GRUs on some
text tasks.

Usage
~~~~~
1. **Register** it in `src/models/__init__.py` (see instructions in chat).
2. Put `MODEL_ARCH = "lstm_rnn"` in `src/config.py`.
3. Optional hyper-params in `Config`: `HIDDEN_DIM`, `NUM_LAYERS`,
   `BIDIRECTIONAL`, `DROPOUT`.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

__all__ = ["LSTMRNN", "LstmRnn", "LSTMClassifier"]


class LSTMRNN(nn.Module):
    """Bidirectional LSTM network for sentence-level classification."""

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
        **_: Any,  # swallow unexpected kwargs for compatibility
    ) -> None:
        super().__init__()

        # Store meta-data for easy access / checkpointing
        self.vocab_size = vocab_size
        self.num_classes = num_classes

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)
        fc_in = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(fc_in, num_classes)

        self._init_weights()

    # ------------------------------------------------------------------ #
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Compute logits for a batch.

        Parameters
        ----------
        input_ids : LongTensor of shape ``(batch, seq_len)``
            Padded token id sequences.
        """
        x = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        output, (h_n, c_n) = self.lstm(x)

        # h_n → (num_layers * num_directions, batch, hidden_dim)
        if self.lstm.bidirectional:
            h_fwd = h_n[-2]
            h_bwd = h_n[-1]
            h = torch.cat((h_fwd, h_bwd), dim=1)
        else:
            h = h_n[-1]

        logits = self.fc(self.dropout(h))
        return logits

    # ------------------------------------------------------------------ #
    def _init_weights(self) -> None:
        # Embedding: small uniform range
        nn.init.uniform_(self.embedding.weight, -0.05, 0.05)

        # LSTM params: Xavier/orthogonal initialisation
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
                # LSTM forget gate bias trick
                n = param.data.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)


# Backwards-compat aliases
LstmRnn = LSTMRNN  # pylint: disable=invalid-name
LSTMClassifier = LSTMRNN
