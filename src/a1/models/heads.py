"""The word tables every neural model shares (D0.6 part 2, proposed 12 Oct).

Each model reads words through an input table E (|V| x d, d = 100) and scores
the next word through an output head:

    top hidden state (size H) -> projection to OUTPUT_DIM numbers -> output layer -> |V| scores

The projection belongs to the model's core; the output layer (OUTPUT_DIM x |V|
weights plus |V| biases) has the same size in every model, so the two
vocabulary-sized tables are identical across models and only the core differs.
A projection before the output layer keeps that layer small (Sak, Senior and
Beaufays, 2014; Jozefowicz et al., 2016).

The output layer is not tied to E: in Part II, E can be frozen (GloVe), and a
tied output layer would then be frozen too.
"""

from __future__ import annotations

import torch
from torch import nn

from a1.data.vocab import PAD_ID

EMBEDDING_DIM = 100  # d, locked in D2.2 for every input embedding
OUTPUT_DIM = 100  # width of the vector that feeds the output layer
EMBEDDING_INIT = 0.1  # E starts uniform in [-0.1, 0.1], on the scale of the output layer


def make_embedding(vocab_size: int, dim: int = EMBEDDING_DIM) -> nn.Embedding:
    """The input table E, with the <pad> row fixed at zero."""
    embedding = nn.Embedding(vocab_size, dim, padding_idx=PAD_ID)
    with torch.no_grad():
        embedding.weight.uniform_(-EMBEDDING_INIT, EMBEDDING_INIT)
        embedding.weight[PAD_ID].zero_()
    return embedding


class OutputHead(nn.Module):
    """Projection from the model's top hidden state to OUTPUT_DIM, then |V| scores."""

    def __init__(self, hidden: int, vocab_size: int, output_dim: int = OUTPUT_DIM) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden, output_dim)
        self.output = nn.Linear(output_dim, vocab_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.output(self.projection(hidden))
