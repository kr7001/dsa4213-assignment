"""Tests for the training rules, the noise scale and scoring."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from a1.data.sequences import EncodedReviews
from a1.data.vocab import PAD_ID
from a1.evaluation.perplexity import evaluate
from a1.evaluation.timing import training_speed
from a1.models.build import build_model
from a1.training.noise import measure_noise_scale, simple_noise_scale
from a1.training.schedule import lr_factor
from a1.training.stopping import EarlyStopping
from a1.training.trainer import TrainConfig, gradient_norm_summary, make_optimizer, train

CPU = torch.device("cpu")


def test_learning_rate_plan() -> None:
    total = 1000  # warm-up is 4% = 40 steps
    factors = [lr_factor(step, total) for step in range(total)]
    assert factors[0] == pytest.approx(1 / 40) and factors[39] == pytest.approx(1.0)
    assert factors[40] == pytest.approx(1.0)
    assert factors[-1] == pytest.approx(0.1)  # 10% of the peak exactly at the last step
    after = np.array(factors[40:])
    assert np.all(np.diff(after) <= 1e-12)


def test_early_stopping_waits_three_checks() -> None:
    stopper = EarlyStopping(patience=3)
    for step, value in enumerate([5.0, 4.0, 4.1, 3.9, 4.0, 4.0, 3.95]):
        stopper.update(value, step)
    assert stopper.best == 3.9 and stopper.best_step == 3 and stopper.should_stop


def test_weight_decay_skips_biases_norms_and_the_input_table() -> None:
    model = build_model(
        {"family": "transformer", "width": 16, "layers": 1, "heads": 2, "feedforward": 16}, 30
    )
    optimizer = make_optimizer(model, TrainConfig())
    decayed = {id(p) for p in optimizer.param_groups[0]["params"]}
    names = {name: id(p) in decayed for name, p in model.named_parameters()}
    assert not names["embedding.weight"] and not names["final_norm.weight"]
    assert not names["head.output.bias"] and names["head.output.weight"]
    assert names["blocks.0.qkv.weight"] and optimizer.param_groups[1]["weight_decay"] == 0.0


def test_noise_scale_estimator_recovers_a_known_value() -> None:
    rng = np.random.default_rng(1)
    dim, sigma, small, k = 200, 3.0, 64, 8
    true_gradient = rng.normal(size=dim)
    expected = dim * sigma**2 / float(true_gradient @ true_gradient)  # tr(Sigma) / |G|^2
    estimates = []
    for _ in range(300):
        per_target = true_gradient + sigma * rng.normal(size=(small * k, dim))
        micro = per_target.reshape(k, small, dim).mean(axis=1)
        big = per_target.mean(axis=0)
        result = simple_noise_scale((micro**2).sum(axis=1), float(big @ big), small, small * k)
        estimates.append((result.trace, result.gradient_sq))
    trace, gradient_sq = np.mean(estimates, axis=0)  # ratio of averages, as McCandlish et al. do
    assert trace / gradient_sq == pytest.approx(expected, rel=0.1)


def _toy_reviews(n: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    # A predictable language: each review counts up from a random start.
    reviews = []
    for _ in range(n):
        start, length = rng.integers(4, 20), rng.integers(3, 12)
        reviews.append([4 + (start - 4 + i) % 26 for i in range(length)])
    return reviews


def test_short_training_run_learns_and_reports() -> None:
    train_data, valid_data = (
        EncodedReviews.from_lists(_toy_reviews(400, 0)),
        EncodedReviews.from_lists(_toy_reviews(60, 1)),
    )
    model = build_model({"family": "lstm", "hidden": 32}, 30)
    before = evaluate(model, valid_data, 512, CPU)
    config = TrainConfig(peak_lr=1e-2, batch_tokens=512, epochs=3, log_every=10, checks_per_epoch=2)
    calls: list[int] = []
    summary = train(
        model,
        train_data,
        valid_data,
        config,
        CPU,
        after_step=lambda step, _: calls.append(step),
        log=lambda _: None,
    )
    after = evaluate(model, valid_data, 512, CPU)
    assert after["J"] < before["J"] - 0.5
    assert summary["best_valid_J"] == pytest.approx(after["J"], rel=1e-5)
    assert calls == list(range(1, summary["steps"] + 1))
    norms = summary["gradient_norms"]
    assert norms["steps"] == summary["steps"] and 0.0 <= norms["share_clipped"] <= 1.0
    estimate = measure_noise_scale(model, train_data, 256, 8, CPU, seed=3)
    assert math.isfinite(estimate.b_simple)


def test_a_uniform_model_scores_ppl_equal_to_the_vocabulary_size() -> None:
    vocab_size = 30

    class Uniform(nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return torch.zeros(*inputs.shape, vocab_size)

    reviews = EncodedReviews.from_lists([[5, 6, 7], [8], [9, 10]])
    result = evaluate(Uniform(), reviews, 64, CPU, by_position=True)
    assert result["targets"] == 9 and result["PPL"] == pytest.approx(vocab_size)
    assert result["targets_by_position"][:4] == [3, 3, 2, 1]
    assert result["unk_share"] == 0.0 and PAD_ID == 0


def test_gradient_summary_flags_the_safe_range() -> None:
    summary = gradient_norm_summary([0.4, 0.5, 0.6, 3.0], clip_norm=1.0)
    assert summary["share_clipped"] == 0.25 and summary["threshold_in_safe_range"]
    assert not gradient_norm_summary([30.0, 40.0], clip_norm=1.0)["threshold_in_safe_range"]


def test_speed_is_timed_on_a_second_pass() -> None:
    reviews = EncodedReviews.from_lists(_toy_reviews(200, 2))
    model = build_model({"family": "rnn", "hidden": 16}, 30)
    row = training_speed(model, make_optimizer(model, TrainConfig()), reviews, 256, CPU, 3, 0)
    assert row["steps_timed"] == 3 and row["targets_per_second"] > 0
    assert 0.0 <= row["padding_share"] < 1.0


def test_tuning_keeps_stepping_past_an_edge_until_the_best_rate_is_inside() -> None:
    from a1.training.tuning import best_learning_rate, next_learning_rate

    def search(j: dict[float, float]) -> tuple[list[float], float]:
        tries: dict[float, float] = {}
        order = []
        while (rate := next_learning_rate(tries)) is not None:
            order.append(rate)
            tries[rate] = j[rate]
        return order, best_learning_rate(tries)

    # Best at the top edge twice, then a worse rate: the best ends up inside.
    order, best = search({3e-4: 5.0, 1e-3: 4.8, 3e-3: 4.7, 1e-2: 4.6, 3e-2: 4.65})
    assert order == [3e-4, 1e-3, 3e-3, 1e-2, 3e-2] and best == 1e-2
    # A diverged try ends the search on its side.
    order, best = search({3e-4: 5.0, 1e-3: 4.8, 3e-3: 4.7, 1e-2: math.inf})
    assert order == [3e-4, 1e-3, 3e-3, 1e-2] and best == 3e-3
    # Best inside the starting grid: no extra tries.
    order, best = search({3e-4: 5.0, 1e-3: 4.6, 3e-3: 4.7})
    assert order == [3e-4, 1e-3, 3e-3] and best == 1e-3
    # Best at the low edge: step down until a smaller rate is worse.
    order, best = search({3e-4: 4.5, 1e-3: 4.6, 3e-3: 4.7, 1e-4: 4.4, 3e-5: 4.45})
    assert order == [3e-4, 1e-3, 3e-3, 1e-4, 3e-5] and best == 1e-4
    # The ladder ends at 0.1, so a model that keeps improving stops there.
    j = {3e-4: 5.0, 1e-3: 4.9, 3e-3: 4.8, 1e-2: 4.7, 3e-2: 4.6, 1e-1: 4.5}
    order, best = search(j)
    assert order == [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1] and best == 1e-1


def test_training_sample_is_fixed_and_just_large_enough() -> None:
    from a1.data.sequences import training_sample

    reviews = EncodedReviews.from_lists(_toy_reviews(300, 4))
    sample = training_sample(reviews, 500, seed=4213)
    again = training_sample(reviews, 500, seed=4213)
    assert np.array_equal(sample.review_ids, again.review_ids)
    assert np.all(np.diff(sample.review_ids) > 0)  # kept in corpus order
    assert sample.n_targets >= 500
    smallest = int(sample.lengths.min()) + 1
    assert sample.n_targets - smallest < 500 + (int(reviews.lengths.max()) + 1)


def test_checks_score_the_training_sample_and_split_the_time() -> None:
    from a1.data.sequences import training_sample

    train_data = EncodedReviews.from_lists(_toy_reviews(300, 5))
    valid_data = EncodedReviews.from_lists(_toy_reviews(40, 6))
    sample = training_sample(train_data, valid_data.n_targets, seed=4213)
    model = build_model({"family": "rnn", "hidden": 16}, 30)
    config = TrainConfig(peak_lr=1e-2, batch_tokens=512, epochs=2, checks_per_epoch=2)
    summary = train(
        model, train_data, valid_data, config, CPU, log=lambda _: None, train_sample=sample
    )
    checks = [row for row in summary["history"] if "valid_J" in row]
    assert checks and all(math.isfinite(row["train_sample_J"]) for row in checks)
    assert summary["eval_seconds"] > 0
    assert summary["train_seconds"] == pytest.approx(summary["seconds"] - summary["eval_seconds"])
    assert 0 < summary["train_seconds_to_best"] <= summary["train_seconds"]
