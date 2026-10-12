"""Tests for the n-gram models: against a slow reference, sums to one, and counts."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pytest

from a1.data.sequences import EncodedReviews
from a1.data.vocab import BOS_ID, EOS_ID
from a1.models.ngram import AddK, KneserNey, NgramCounts, discounts, unseen_shares

VOCAB = 12  # ids 0-3 are special, 4-11 are words


def toy_reviews(n: int, seed: int) -> EncodedReviews:
    rng = np.random.default_rng(seed)
    reviews = []
    for _ in range(n):
        start = int(rng.integers(4, 9))
        length = int(rng.integers(1, 7))
        # mostly counting upwards, with some noise, so higher orders help
        review = [
            int(rng.integers(4, 12)) if rng.random() < 0.2 else 4 + (start - 4 + i) % 8
            for i in range(length)
        ]
        reviews.append(review)
    return EncodedReviews.from_lists(reviews)


def sequences(reviews: EncodedReviews) -> list[list[int]]:
    return [[BOS_ID, *reviews.review(i).tolist(), EOS_ID] for i in range(len(reviews))]


def reference_kneser_ney(train: EncodedReviews, order: int):
    """A slow, dictionary version of interpolated modified Kneser-Ney, for checking."""
    seqs = sequences(train)
    raw = defaultdict(Counter)
    for m in range(1, order + 1):
        for seq in seqs:
            for end in range(m - 1, len(seq)):
                raw[m][tuple(seq[end - m + 1 : end + 1])] += 1
    adjusted = {}
    for m in range(1, order + 1):
        if m == order:
            counts = Counter(raw[m])
        else:
            left = Counter(gram[1:] for gram in raw[m + 1])
            counts = Counter({g: (raw[m][g] if g[0] == BOS_ID else left[g]) for g in raw[m]})
        if m == 1:
            counts[(BOS_ID,)] = 0
        adjusted[m] = counts
    d = {m: discounts(np.array(list(adjusted[m].values()), dtype=float)) for m in adjusted}

    def disc(m: int, c: float) -> float:
        return d[m][min(int(c), 3)]

    def prob(history: tuple, word: int, m: int) -> float:
        if m == 1:
            c1 = adjusted[1]
            total = sum(c1.values())
            gamma = sum(disc(1, c) for c in c1.values() if c > 0) / total
            c = c1.get((word,), 0)
            return max(c - disc(1, c), 0) / total + gamma / VOCAB
        lower = prob(history[1:], word, m - 1)
        if len(history) < m - 1:
            return prob(history, word, m - 1)
        h = history[-(m - 1) :]
        row = {g[-1]: c for g, c in adjusted[m].items() if g[:-1] == h}
        total = sum(row.values())
        if total == 0:
            return prob(h[1:], word, m - 1)
        gamma = sum(disc(m, c) for c in row.values() if c > 0) / total
        c = row.get(word, 0)
        return max(c - disc(m, c), 0) / total + gamma * prob(h[1:], word, m - 1) + 0 * lower

    return prob


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_kneser_ney_matches_a_slow_reference(order: int) -> None:
    train, test = toy_reviews(150, 0), toy_reviews(25, 1)
    table = NgramCounts.build(train, VOCAB, max_order=4)
    model = KneserNey(table, order)
    fast = np.exp(model.log_probs(test))
    prob = reference_kneser_ney(train, order)
    slow = []
    for seq in sequences(test):
        for t in range(1, len(seq)):
            history = tuple(seq[max(0, t - order + 1) : t])
            slow.append(prob(history, seq[t], order))
    assert fast.size == test.n_targets
    assert np.allclose(fast, slow, rtol=1e-10, atol=1e-12)


def test_kneser_ney_never_gives_zero_probability_on_repetitive_text() -> None:
    # Every review is the same few words, so higher orders have no singletons.
    train = EncodedReviews.from_lists([[4, 5, 6, 7]] * 50 + [[4, 5, 6, 8]] * 50)
    test = EncodedReviews.from_lists([[4, 5, 9, 7], [11]])
    table = NgramCounts.build(train, VOCAB, max_order=4)
    for order in (1, 2, 3, 4):
        assert np.all(np.isfinite(KneserNey(table, order).log_probs(test)))


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_every_history_gives_a_distribution_that_sums_to_one(order: int) -> None:
    train = toy_reviews(150, 0)
    table = NgramCounts.build(train, VOCAB, max_order=4)
    add_k = AddK(table, order, k=0.1)
    add_k.fit_weights(toy_reviews(30, 2))
    for model in (KneserNey(table, order), add_k):
        for history in ([BOS_ID], [BOS_ID, 5], [BOS_ID, 5, 6, 7], [BOS_ID, 11, 4, 11, 4]):
            p = model.distribution(history)
            assert p.shape == (VOCAB,) and np.all(p > 0)
            assert p.sum() == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("model_name", ["kn", "addk"])
def test_scores_agree_with_the_full_distribution(model_name: str) -> None:
    train, test = toy_reviews(150, 0), toy_reviews(10, 3)
    table = NgramCounts.build(train, VOCAB, max_order=3)
    model = KneserNey(table, 3) if model_name == "kn" else AddK(table, 3, k=0.01)
    scores = np.exp(model.log_probs(test))
    expected = [
        model.distribution(seq[:t])[seq[t]] for seq in sequences(test) for t in range(1, len(seq))
    ]
    assert np.allclose(scores, expected, rtol=1e-10)


def test_add_k_weights_raise_validation_likelihood() -> None:
    train, valid = toy_reviews(150, 0), toy_reviews(40, 4)
    table = NgramCounts.build(train, VOCAB, max_order=3)
    model = AddK(table, 3, k=0.01)
    before = model.log_probs(valid).mean()
    model.fit_weights(valid)
    assert model.weights.sum() == pytest.approx(1.0) and np.all(model.weights >= 0)
    assert model.log_probs(valid).mean() >= before


def test_discounts_follow_chen_and_goodman() -> None:
    counts = np.array([1] * 10 + [2] * 4 + [3] * 2 + [4] * 1, dtype=float)
    y = 10 / (10 + 2 * 4)
    expected = [0.0, 1 - 2 * y * 4 / 10, 2 - 3 * y * 2 / 4, 3 - 4 * y * 1 / 2]
    assert np.allclose(discounts(counts), expected)
    # Without singletons the formula fails; the KenLM fallback discounts take over.
    assert np.allclose(discounts(np.array([3.0, 5.0, 7.0])), [0.0, 0.5, 1.0, 1.5])


def test_counts_and_unseen_shares() -> None:
    train = EncodedReviews.from_lists([[4, 5], [4, 6]])
    table = NgramCounts.build(train, VOCAB, max_order=2)
    # Distinct bigrams: (bos,4), (4,5), (5,eos), (4,6), (6,eos).
    assert table.targets == 6 and table.stored_entries(2) == table.keys[1].size + 5
    # Continuation count of "4": it follows only <bos>; "5" follows only "4".
    unigram = dict(zip(table.keys[1].tolist(), table.continuation[1].tolist(), strict=True))
    assert unigram[4] == 1 and unigram[5] == 1 and unigram[EOS_ID] == 2
    test = EncodedReviews.from_lists([[4, 7]])
    shares = unseen_shares(table, test, 2)
    # Targets 4, 7, <eos>: the word 7 never occurred; bigrams (bos,4) seen, (4,7) and (7,eos) not.
    assert shares[1] == pytest.approx(1 / 3) and shares[2] == pytest.approx(2 / 3)
