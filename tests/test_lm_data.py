"""Tests for the vocabulary, piles, batches and padding."""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from a1.data.sequences import (
    EncodedReviews,
    Pile,
    collate,
    count_tokens,
    encode_pile,
    make_batches,
    padded_width,
)
from a1.data.vocab import BOS_ID, EOS_ID, PAD_ID, SPECIAL_TOKENS, UNK_ID, Vocabulary

reviews_from = EncodedReviews.from_lists


def test_vocabulary_order_threshold_and_round_trip(tmp_path: Path) -> None:
    counts = Counter({"the": 9, "good": 5, "bad": 5, "rare": 4, "<num>": 6})
    vocab = Vocabulary.build(counts, min_count=5)
    assert vocab.tokens[:4] == SPECIAL_TOKENS
    # Most frequent first, ties alphabetical; "rare" (4 < 5) is left out.
    assert vocab.tokens[4:] == ("the", "<num>", "bad", "good")
    assert vocab.encode(["the", "rare", "good"]) == [4, UNK_ID, 7]
    vocab.save(tmp_path / "vocab.json")
    loaded = Vocabulary.load(tmp_path / "vocab.json")
    assert loaded.tokens == vocab.tokens and loaded.min_count == 5
    with pytest.raises(ValueError):
        Vocabulary(["the", *SPECIAL_TOKENS])


def test_piles_follow_the_split_key(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl.gz"
    rows = [(0.10, "love it ."), (0.44, "love this"), (0.46, "hate it"), (0.90, "it is ok")]
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for i, (key, text) in enumerate(rows):
            handle.write(json.dumps({"id": i, "split_key": key, "text": text}) + "\n")
    train, valid = Pile("train", 0.0, 0.45), Pile("valid", 0.45, 0.5)
    counts, n = count_tokens(path, train)
    assert n == 2 and counts["love"] == 2 and "hate" not in counts
    vocab = Vocabulary.build(counts, min_count=1)
    encoded = encode_pile(path, valid, vocab)
    assert encoded.review_ids.tolist() == [2]
    assert encoded.review(0).tolist() == [UNK_ID, vocab.index["it"]]  # "hate" is unseen
    assert encoded.n_targets == 3  # two tokens plus <eos>


def test_batches_cover_every_review_once_within_budget() -> None:
    rng = np.random.default_rng(0)
    lengths = rng.integers(1, 120, size=500)
    for seed in (None, 7):
        batches = make_batches(lengths, max_tokens=1024, seed=seed)
        seen = np.concatenate(batches)
        assert sorted(seen.tolist()) == list(range(500))
        for index in batches:
            assert index.size * padded_width(int(lengths[index].max()) + 1) <= 1024
    same = make_batches(lengths, 1024, seed=7)
    assert all(
        np.array_equal(a, b) for a, b in zip(same, make_batches(lengths, 1024, seed=7), strict=True)
    )
    other = make_batches(lengths, 1024, seed=8)
    assert not all(np.array_equal(a, b) for a, b in zip(same, other, strict=False))
    with pytest.raises(ValueError):
        make_batches(np.array([2000]), max_tokens=1024)


def test_collate_shifts_inputs_and_targets() -> None:
    reviews = reviews_from([[10, 11, 12], [20]])
    inputs, targets = collate(reviews, np.array([0, 1]), fixed_widths=False)
    assert inputs.tolist() == [[BOS_ID, 10, 11, 12], [BOS_ID, 20, PAD_ID, PAD_ID]]
    assert targets.tolist() == [[10, 11, 12, EOS_ID], [20, EOS_ID, PAD_ID, PAD_ID]]
    assert reviews.n_targets == 6
    subset = reviews.subset(np.array([1]))
    assert subset.review(0).tolist() == [20] and subset.n_targets == 2


def test_batches_are_padded_to_a_few_fixed_widths() -> None:
    assert [padded_width(n) for n in (1, 9, 33, 65, 449, 513)] == [8, 16, 40, 80, 513, 513]
    assert padded_width(9, fixed=False) == 9
    reviews = reviews_from([[10, 11, 12], [20]])
    inputs, targets = collate(reviews, np.array([0, 1]))
    assert inputs.shape == targets.shape == (2, 8)
    assert int((targets != PAD_ID).sum()) == reviews.n_targets  # padding is never a target
