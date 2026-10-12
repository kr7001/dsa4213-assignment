"""Tests for the three neural models: sizes, causality and padding."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from a1.data.sequences import EncodedReviews, collate
from a1.data.vocab import BOS_ID, PAD_ID
from a1.models.build import build_model, count_parameters
from a1.models.heads import EMBEDDING_DIM, OUTPUT_DIM

VOCAB = 50
CONFIGS = {
    "rnn": {"family": "rnn", "hidden": 24, "layers": 2},
    "lstm": {"family": "lstm", "hidden": 12, "layers": 1},
    "transformer": {
        "family": "transformer",
        "width": 16,
        "layers": 2,
        "heads": 4,
        "feedforward": 32,
    },
}


def test_word_tables_are_identical_and_core_counts_follow_the_formulas() -> None:
    counts = {name: count_parameters(build_model(c, VOCAB)) for name, c in CONFIGS.items()}
    tables = VOCAB * EMBEDDING_DIM + OUTPUT_DIM * VOCAB + VOCAB
    assert all(c["word_tables"] == tables for c in counts.values())
    d, out = EMBEDDING_DIM, OUTPUT_DIM
    h = 24  # two-layer RNN: W_x, W_h and two biases per layer, then the projection
    rnn = (h * d + h * h + 2 * h) + (h * h + h * h + 2 * h) + (h * out + out)
    assert counts["rnn"]["core"] == rnn
    h = 12  # one-layer LSTM: four gates
    assert counts["lstm"]["core"] == 4 * (h * d + h * h + 2 * h) + (h * out + out)
    assert all(
        c["total"] == c["word_tables"] + c["core"] == c["trainable"] for c in counts.values()
    )


@pytest.mark.parametrize("name", list(CONFIGS))
def test_no_position_sees_the_future(name: str) -> None:
    torch.manual_seed(0)
    model = build_model(CONFIGS[name], VOCAB).eval()
    tokens = torch.randint(4, VOCAB, (2, 12))
    tokens[:, 0] = BOS_ID
    changed = tokens.clone()
    changed[:, 7:] = torch.randint(4, VOCAB, (2, 5))
    with torch.no_grad():
        before, after = model(tokens), model(changed)
    assert torch.allclose(before[:, :7], after[:, :7], atol=1e-6)
    assert not torch.allclose(before[:, 7:], after[:, 7:])


@pytest.mark.parametrize("name", list(CONFIGS))
def test_padding_does_not_change_the_loss(name: str) -> None:
    torch.manual_seed(0)
    model = build_model(CONFIGS[name], VOCAB).eval()
    reviews = EncodedReviews.from_lists([[5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]])

    def summed_loss(index: list[int]) -> torch.Tensor:
        inputs, targets = collate(reviews, np.array(index), fixed_widths=False)
        with torch.no_grad():
            logits = model(inputs)
        losses = F.cross_entropy(
            logits.reshape(-1, VOCAB), targets.reshape(-1), ignore_index=PAD_ID, reduction="none"
        ).view(targets.shape)
        return losses[0].sum()

    # Review 0 alone has no padding; next to the longer review it gets 5 pad positions.
    assert torch.allclose(summed_loss([0]), summed_loss([0, 1]), atol=1e-5)


def test_transformer_context_and_lstm_forget_bias() -> None:
    model = build_model(CONFIGS["transformer"], VOCAB)
    with pytest.raises(ValueError):
        model(torch.zeros(1, 514, dtype=torch.long))
    lstm = build_model(CONFIGS["lstm"], VOCAB)
    h = CONFIGS["lstm"]["hidden"]
    assert torch.all(lstm.rnn.bias_ih_l0[h : 2 * h] == 1.0)
    assert torch.all(lstm.rnn.bias_hh_l0[h : 2 * h] == 0.0)
    assert torch.all(lstm.embedding.weight[PAD_ID] == 0)
