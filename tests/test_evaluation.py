"""Tests for the final evaluation: loss by position, FLOPs, timing and generated text."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from a1.data.sequences import EncodedReviews
from a1.data.vocab import BOS_ID, EOS_ID, PAD_ID
from a1.evaluation.flops import forward_flops
from a1.evaluation.perplexity import evaluate
from a1.evaluation.positions import (
    group_positions,
    mean_context,
    neural_position_sums,
    ngram_position_sums,
)
from a1.evaluation.timing import ngram_scoring_speed, scoring_speed
from a1.generation import (
    bootstrap_scores,
    choose_temperature,
    detokenize,
    distinct_halves,
    distinct_share,
    generate,
    make_sampler,
    middle_sample,
    repeated_share,
    seen_in_training,
    temperature_probabilities,
    text_scores,
    zero_crossing,
)
from a1.models.build import build_model
from a1.models.heads import OUTPUT_DIM
from a1.models.ngram import KneserNey, NgramCounts

VOCAB = 30  # 4 special tokens + 26 words


def toy_reviews(count: int, seed: int, low: int = 3, high: int = 15) -> EncodedReviews:
    rng = np.random.default_rng(seed)
    reviews = []
    for _ in range(count):
        n = int(rng.integers(low, high))
        # a skewed word distribution, so some n-grams repeat
        words = np.minimum(4 + rng.geometric(0.25, size=n) - 1, VOCAB - 1)
        reviews.append(words.tolist())
    return EncodedReviews.from_lists(reviews)


def test_fast_kneser_ney_distribution_matches_the_full_one() -> None:
    train = toy_reviews(300, seed=1)
    table = NgramCounts.build(train, VOCAB, 5)
    model = KneserNey(table, 5)
    rng = np.random.default_rng(2)
    for _ in range(40):
        history = [BOS_ID, *rng.integers(4, VOCAB, size=int(rng.integers(0, 9))).tolist()]
        slow = model.distribution(history)
        fast = model.next_distribution(history)
        np.testing.assert_allclose(fast, slow, rtol=1e-12, atol=1e-15)
        assert fast.sum() == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(
    "spec",
    [
        {"family": "rnn", "hidden": 12},
        {"family": "lstm", "hidden": 12},
        {"family": "transformer", "width": 100, "layers": 2, "heads": 4, "feedforward": 16},
    ],
)
def test_samplers_give_the_same_probabilities_as_the_full_forward(spec: dict) -> None:
    torch.manual_seed(0)
    model = build_model(spec, VOCAB).eval()
    sampler = make_sampler(model, torch.device("cpu"))
    prompts = np.array([[BOS_ID, 5, 6, 7], [BOS_ID, 8, 9, 10]])
    log_p = sampler.start(prompts)
    tokens = torch.as_tensor(prompts)
    for step in range(3):
        with torch.no_grad():
            full = torch.log_softmax(model(tokens), dim=-1)[:, -1].double().numpy()
        np.testing.assert_allclose(log_p, full, atol=1e-5)
        nxt = np.array([11 + step, 12 + step])
        log_p = sampler.advance(nxt)
        tokens = torch.cat([tokens, torch.as_tensor(nxt)[:, None]], dim=1)


def test_temperature_matches_the_lecture_example() -> None:
    # Lecture1_Demo / Lecture 3: p = (0.6, 0.3, 0.1) at T = 0.5 -> (0.783, 0.196, 0.022)
    log_p = np.log(np.array([[1e-12, 1e-12, 1e-12, 1e-12, 0.6, 0.3, 0.1]]))
    q = temperature_probabilities(log_p, 0.5)
    np.testing.assert_allclose(q[0, 4:], [0.7826, 0.1957, 0.0217], atol=1e-4)
    assert q[0, [PAD_ID, BOS_ID, EOS_ID]].sum() == 0.0  # never written
    unchanged = temperature_probabilities(log_p, 1.0)[0, 4:]
    np.testing.assert_allclose(unchanged, [0.6, 0.3, 0.1], atol=1e-9)


class FixedSampler:
    """Every step gives the same distribution, which puts most mass on blocked tokens."""

    def __init__(self, probabilities: np.ndarray) -> None:
        self.log_p = np.log(probabilities)

    def start(self, prompts: np.ndarray) -> np.ndarray:
        return np.tile(self.log_p, (prompts.shape[0], 1))

    def advance(self, tokens: np.ndarray) -> np.ndarray:
        return np.tile(self.log_p, (tokens.shape[0], 1))


def test_generation_never_writes_blocked_tokens_and_repeats_with_a_seed() -> None:
    p = np.full(VOCAB, 0.01)
    p[[PAD_ID, BOS_ID, EOS_ID]] = 0.3
    p /= p.sum()
    sampler = FixedSampler(p)
    prompts = np.full((50, 1), BOS_ID)
    tokens, own = generate(sampler, prompts, 20, 1.0, seed=4213)
    assert tokens.shape == (50, 20)
    assert not np.isin(tokens, [PAD_ID, BOS_ID, EOS_ID]).any()
    np.testing.assert_allclose(own, np.log(p[tokens]))  # log P under the model as trained
    again, _ = generate(sampler, prompts, 20, 1.0, seed=4213)
    assert (again == tokens).all()
    other, _ = generate(sampler, prompts, 20, 1.0, seed=4214)
    assert (other != tokens).any()


def test_repetition_variety_and_middle_sample() -> None:
    assert repeated_share(np.array([[1, 2, 1, 2, 1]]), 2)[0] == pytest.approx(0.5)
    assert repeated_share(np.array([[1, 2, 3, 4]]), 4)[0] == 0.0
    written = np.array([[1, 2, 3], [1, 2, 4]])
    assert distinct_share(written, 1) == pytest.approx(4 / 6)
    assert distinct_share(written, 2) == pytest.approx(3 / 6)
    own = -np.array([[1.0], [5.0], [3.0], [2.0], [4.0]])  # perplexity order: 0, 3, 2, 4, 1
    assert middle_sample(own) == 2
    with pytest.raises(ValueError):
        middle_sample(own[:4])


def test_copied_windows_are_found_in_the_training_reviews() -> None:
    train = EncodedReviews.from_lists([[5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16]])
    table = NgramCounts.build(train, VOCAB, 5)
    texts = np.array(
        [
            [BOS_ID, 5, 6, 7, 8, 9, 10],  # both written windows come from review 1
            [BOS_ID, 5, 6, 7, 8, 9, 16],  # the last window is new
        ]
    )
    seen = seen_in_training(table, texts, first=5)
    assert seen.tolist() == [[1.0, 1.0], [1.0, 0.0]]
    scores = text_scores(table, texts[:, :5], texts[:, 5:])
    assert scores["copied_5gram_share"] == pytest.approx(0.75)
    # After a short prompt, the first written tokens have no full 5-token window yet:
    # those positions are left out, not counted as new.
    short = np.array([[BOS_ID, 5, 6, 7, 8, 9]])
    early = seen_in_training(table, short, first=1)
    assert np.isnan(early[0, :3]).all() and early[0, 3:].tolist() == [1.0, 1.0]
    assert text_scores(table, short[:, :1], short[:, 1:])["copied_5gram_share"] == 1.0


def test_intervals_cover_the_point_and_distinct_n_uses_halves() -> None:
    train = toy_reviews(300, seed=7, low=12, high=20)
    table = NgramCounts.build(train, VOCAB, 5)
    texts = np.stack([np.r_[BOS_ID, train.review(i)[:12]] for i in range(60)]).astype(np.int64)
    prompts, written = texts[:, :3], texts[:, 3:]
    point = text_scores(table, prompts, written)
    intervals = bootstrap_scores(table, prompts, written, resamples=200, seed=1)
    assert set(intervals) == {"seq_rep_2", "seq_rep_4", "copied_5gram_share", "unk_share"}
    for key, (low, high) in intervals.items():
        assert low <= point[key] + 1e-12 and point[key] <= high + 1e-12
    halves = distinct_halves(written)
    assert len(halves["distinct_2"]) == 2


def test_nothing_converts_to_float64_on_the_gpu() -> None:
    # MPS has no float64: tensors must be moved to the CPU before .double().
    root = Path(__file__).resolve().parents[1]
    for path in [*root.glob("src/a1/**/*.py"), *root.glob("scripts/*.py")]:
        assert ".double().cpu()" not in path.read_text(), path


def test_temperature_choice_and_zero_crossings() -> None:
    grid = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    gaps = {
        "a": [t - 0.9 for t in grid],
        "b": [3 * (t - 0.9) for t in grid],
        "c": [2 * (t - 0.95) for t in grid],
    }
    choice = choose_temperature(grid, gaps)
    assert choice["temperature"] == pytest.approx(0.9)
    assert choice["own_temperature"]["c"] == pytest.approx(0.95)
    assert not choice["at_low_edge"] and not choice["at_high_edge"]
    rising = choose_temperature(grid, {"a": [t + 1.0 for t in grid]})
    assert rising["at_low_edge"] and rising["temperature"] == pytest.approx(0.6)
    assert zero_crossing([0.5, 1.0], [1.0, 2.0]) is None


def test_positions_are_grouped_by_their_targets() -> None:
    sums = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    counts = np.array([1, 1, 1, 1, 2])
    groups = group_positions(sums, counts, groups=((1, 4), (5, 8)))
    assert groups[0] == {"positions": [1, 4], "targets": 4, "J": 2.5}
    assert groups[1]["J"] == pytest.approx(5.0)


def test_ngram_and_neural_position_sums_agree_with_the_totals() -> None:
    train = toy_reviews(200, seed=3)
    test = toy_reviews(30, seed=4)
    model = KneserNey(NgramCounts.build(train, VOCAB, 3), 3)
    log_probs = model.log_probs(test)
    sums, counts = ngram_position_sums(log_probs, test)
    assert counts.sum() == test.n_targets
    assert sums.sum() == pytest.approx(-log_probs.sum())
    torch.manual_seed(0)
    lstm = build_model({"family": "lstm", "hidden": 8}, VOCAB)
    result = evaluate(lstm, test, 256, torch.device("cpu"), by_position=True)
    n_sums, n_counts = neural_position_sums(result)
    assert n_counts.sum() == test.n_targets
    assert n_sums.sum() / n_counts.sum() == pytest.approx(result["J"], rel=1e-6)


def test_mean_context_counts_bos_and_earlier_words() -> None:
    reviews = EncodedReviews.from_lists([[5], [5, 6]])
    # targets: review 1 at positions 1, 2; review 2 at 1, 2, 3 -> (1 + 2 + 1 + 2 + 3) / 5
    assert mean_context(reviews) == pytest.approx(9 / 5)


def test_flops_count_every_weight_matrix_once() -> None:
    rnn = build_model({"family": "rnn", "hidden": 16}, VOCAB)
    flops = forward_flops(rnn, mean_context=10.0)
    core = 100 * 16 + 16 * 16 + 16 * OUTPUT_DIM
    assert flops["core"] == 2 * core
    assert flops["output_layer"] == 2 * OUTPUT_DIM * VOCAB
    assert flops["attention"] == 0
    lstm = build_model({"family": "lstm", "hidden": 16}, VOCAB)
    assert forward_flops(lstm, 10.0)["core"] == 2 * (4 * (100 * 16 + 16 * 16) + 16 * OUTPUT_DIM)
    spec = {"family": "transformer", "width": 100, "layers": 2, "heads": 4, "feedforward": 16}
    transformer = build_model(spec, VOCAB)
    t_flops = forward_flops(transformer, mean_context=10.0)
    per_layer = 100 * 300 + 100 * 100 + 100 * 16 + 16 * 100
    assert t_flops["core"] == 2 * (2 * per_layer + 100 * OUTPUT_DIM)
    assert t_flops["attention"] == 2 * (2 * 2 * 100 * 10.0)


def test_scoring_speed_reports_the_median_and_the_same_j() -> None:
    torch.manual_seed(0)
    model = build_model({"family": "lstm", "hidden": 8}, VOCAB)
    reviews = toy_reviews(40, seed=5)
    speed = scoring_speed(model, reviews, 256, torch.device("cpu"), repeats=3)
    reference = evaluate(model, reviews, 256, torch.device("cpu"))
    assert speed["J"] == pytest.approx(reference["J"], rel=1e-5)
    assert len(speed["seconds"]) == 3 and speed["targets"] == reviews.n_targets
    low, high = speed["range_targets_per_second"]
    assert low <= speed["targets_per_second"] <= high
    kn = KneserNey(NgramCounts.build(toy_reviews(100, seed=6), VOCAB, 3), 3)
    ngram = ngram_scoring_speed(kn, reviews, repeats=2)
    assert ngram["targets"] == reviews.n_targets


def test_detokenize_attaches_punctuation() -> None:
    assert detokenize(["i", "love", "it", "!", "so", "soft", "."]) == "i love it! so soft."
