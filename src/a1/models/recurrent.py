"""The vanilla RNN and the LSTM language models (Lecture 3).

Both read <bos> w1 ... wn one token at a time. The RNN updates its memory as
h^(t) = tanh(W_h h^(t-1) + W_x x^(t) + b_h); the LSTM adds a gated cell state
c^(t) (Lecture3_Demo, section 7). The state starts at zero for every review,
so no review sees the one before it (D0.5). Padding comes after the last real
token, so it can never change an earlier prediction.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from a1.models.heads import EMBEDDING_DIM, OutputHead, make_embedding

Cell = Literal["rnn", "lstm"]


class RecurrentLM(nn.Module):
    """E -> RNN or LSTM layers -> shared output head.

    Args:
        vocab_size: |V|.
        cell: "rnn" (Elman, tanh) or "lstm".
        hidden: Units per layer, H.
        layers: Stacked recurrent layers.
        dropout: Dropout on the embeddings, between layers and before the head.
        forget_bias: LSTM only: starting value of the forget-gate bias, so the cell
            keeps its memory early in training. Ignored for the RNN.
    """

    def __init__(
        self,
        vocab_size: int,
        cell: Cell,
        hidden: int,
        layers: int = 1,
        dropout: float = 0.0,
        forget_bias: float = 1.0,
        embedding_dim: int = EMBEDDING_DIM,
    ) -> None:
        super().__init__()
        if cell not in ("rnn", "lstm"):
            raise ValueError(f"Unknown cell {cell!r}")
        self.cell = cell
        self.embedding = make_embedding(vocab_size, embedding_dim)
        between = dropout if layers > 1 else 0.0
        if cell == "lstm":
            self.rnn: nn.RNNBase = nn.LSTM(
                embedding_dim, hidden, num_layers=layers, batch_first=True, dropout=between
            )
            with torch.no_grad():  # PyTorch gate order: input, forget, cell, output
                for layer in range(layers):
                    getattr(self.rnn, f"bias_ih_l{layer}")[hidden : 2 * hidden].fill_(forget_bias)
                    getattr(self.rnn, f"bias_hh_l{layer}")[hidden : 2 * hidden].zero_()
        else:
            self.rnn = nn.RNN(
                embedding_dim,
                hidden,
                num_layers=layers,
                nonlinearity="tanh",
                batch_first=True,
                dropout=between,
            )
        self.dropout = nn.Dropout(dropout)
        self.head = OutputHead(hidden, vocab_size)

    def hidden_states(self, inputs: torch.Tensor) -> torch.Tensor:
        """Top-layer states h^(t) for every position, shape (batch, time, H)."""
        states, _ = self.rnn(self.dropout(self.embedding(inputs)))
        return states

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Next-token scores, shape (batch, time, |V|)."""
        return self.head(self.dropout(self.hidden_states(inputs)))
