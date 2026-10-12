"""The decoder-only Transformer language model (Lecture 5).

Each block is pre-LN (Xiong et al., 2020): x + attention(LN(x)), then
x + feed-forward(LN(x)). Attention is causal, so position t only sees
positions up to t (Lecture 5; P1.8), through
scaled_dot_product_attention(..., is_causal=True). Padding comes after the
last real token, so the causal mask alone keeps it out of every real
prediction.

Token vectors from E (d = 100) are mapped to the model width d_model when the
two differ, scaled by sqrt(d_model) and added to sinusoidal position vectors
(Vaswani et al., 2017, sections 3.4 and 3.5). The context L = 513 covers the
longest review, 512 tokens plus <bos> (D0.5).
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from a1.models.heads import EMBEDDING_DIM, OutputHead, make_embedding

CONTEXT = 513  # L, locked in D0.5
Activation = Literal["relu", "gelu"]


def sinusoidal_positions(length: int, width: int) -> torch.Tensor:
    """PE[t, 2i] = sin(t / 10000^(2i/width)), PE[t, 2i+1] = cos(same)."""
    position = torch.arange(length, dtype=torch.float32)[:, None]
    rates = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width))
    table = torch.zeros(length, width)
    table[:, 0::2] = torch.sin(position * rates)
    table[:, 1::2] = torch.cos(position * rates[: width // 2])
    return table


class Block(nn.Module):
    """One pre-LN decoder block: causal multi-head attention, then a feed-forward network."""

    def __init__(
        self, width: int, heads: int, feedforward: int, dropout: float, activation: Activation
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError(f"d_model={width} is not divisible by heads={heads}")
        self.heads = heads
        self.attention_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.attention_out = nn.Linear(width, width)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward_in = nn.Linear(width, feedforward)
        self.feedforward_out = nn.Linear(feedforward, width)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, width = x.shape
        q, k, v = self.qkv(self.attention_norm(x)).split(width, dim=-1)
        shape = (batch, time, self.heads, width // self.heads)
        q, k, v = (part.view(shape).transpose(1, 2) for part in (q, k, v))
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attended = attended.transpose(1, 2).reshape(batch, time, width)
        x = x + F.dropout(self.attention_out(attended), self.dropout, self.training)
        inner = self.activation(self.feedforward_in(self.feedforward_norm(x)))
        inner = F.dropout(inner, self.dropout, self.training)
        return x + F.dropout(self.feedforward_out(inner), self.dropout, self.training)


class TransformerLM(nn.Module):
    """E -> (projection to d_model) + positions -> pre-LN blocks -> final LN -> output head.

    Args:
        vocab_size: |V|.
        width: d_model.
        layers: Number of blocks.
        heads: Attention heads; must divide d_model.
        feedforward: Hidden size of each feed-forward network, d_ff.
        dropout: Dropout on the inputs, attention weights and residual branches.
        activation: "relu" (Vaswani et al., 2017) or "gelu".
        context: Longest input the model accepts, L.
    """

    def __init__(
        self,
        vocab_size: int,
        width: int,
        layers: int,
        heads: int,
        feedforward: int,
        dropout: float = 0.0,
        activation: Activation = "relu",
        context: int = CONTEXT,
        embedding_dim: int = EMBEDDING_DIM,
    ) -> None:
        super().__init__()
        self.context = context
        self.width = width
        self.embedding = make_embedding(vocab_size, embedding_dim)
        self.input_projection = (
            nn.Identity() if width == embedding_dim else nn.Linear(embedding_dim, width)
        )
        self.register_buffer("positions", sinusoidal_positions(context, width), persistent=False)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            Block(width, heads, feedforward, dropout, activation) for _ in range(layers)
        )
        self.final_norm = nn.LayerNorm(width)
        self.head = OutputHead(width, vocab_size)

    def hidden_states(self, inputs: torch.Tensor) -> torch.Tensor:
        """Final-layer states for every position, shape (batch, time, d_model)."""
        time = inputs.size(1)
        if time > self.context:
            raise ValueError(f"Input of {time} positions exceeds the context L = {self.context}")
        tokens = self.input_projection(self.embedding(inputs)) * math.sqrt(self.width)
        x = self.dropout(tokens + self.positions[:time])
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Next-token scores, shape (batch, time, |V|)."""
        return self.head(self.hidden_states(inputs))
