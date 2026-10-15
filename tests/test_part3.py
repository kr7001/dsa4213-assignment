"""Tests for Part III: labelled data, features, the classifier and the statistics."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import stats

from a1.data.sequences import EncodedReviews, Pile
from a1.data.vocab import Vocabulary
from a1.models.build import build_model
from a1.models.heads import OUTPUT_DIM
from a1.part3.classify import (
    ClassifierSettings,
    CountFeatures,
    DenseFeatures,
    best_power,
    choose_power,
    fit_and_test,
    fit_logistic,
    make_folds,
    next_power,
)
from a1.part3.data import (
    NEGATIVE,
    POSITIVE,
    balance,
    copy_checks,
    describe_set,
    group_mask,
    label_of,
    read_part3,
)
from a1.part3.errors import (
    disagreement,
    draw,
    group_accuracy,
    groups,
    has_contrast,
    has_negation,
)
from a1.part3.features import (
    average_vectors,
    length_feature,
    ngram_documents,
    parameter_fingerprint,
    pooled_features,
    token_counts,
    untrained_copy,
)
from a1.part3.stats import (
    agreement_interval,
    bootstrap_accuracies,
    class_scores,
    holm,
    interval,
    mcnemar,
    pair_agreement,
    wilson,
)

VOCAB = 30


def toy_reviews(count: int, seed: int, low: int = 1, high: int = 12) -> EncodedReviews:
    rng = np.random.default_rng(seed)
    return EncodedReviews.from_lists(
        [
            np.minimum(4 + rng.geometric(0.25, size=int(rng.integers(low, high))) - 1, VOCAB - 1)
            .astype(int)
            .tolist()
            for _ in range(count)
        ]
    )


# --- data ------------------------------------------------------------------


def test_labels_follow_d3_1() -> None:
    assert [label_of(s) for s in (1, 2, 3, 4, 5)] == [NEGATIVE, NEGATIVE, None, POSITIVE, POSITIVE]


def write_corpus(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.writelines(json.dumps(row) + "\n" for row in rows)


def corpus_rows() -> list[dict]:
    rows = []
    texts = ["good stuff", "bad stuff .", "love it", "not good at all", "meh"]
    for i in range(400):
        product = f"p{i % 40}"
        key = (i % 40) / 40  # every product keeps one split key
        stars = [1, 2, 3, 4, 5, 5, 5, 4][i % 8]
        rows.append(
            {"id": i, "product": product, "split_key": key, "stars": stars, "text": texts[i % 5]}
        )
    return rows


def test_read_part3_labels_balances_and_checks_copies(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl.gz"
    rows = corpus_rows()
    write_corpus(path, rows)
    vocab = Vocabulary(["<pad>", "<unk>", "<bos>", "<eos>", "good", "stuff", "bad", "."])
    part3 = {
        "part3_train": Pile("part3_train", 0.5, 0.8),
        "part3_test": Pile("part3_test", 0.8, 1.0),
    }
    others = {"lm_train": Pile("lm_train", 0.0, 0.5)}
    reading = read_part3(path, part3, others, vocab)
    assert reading.shared_products == 0
    train = reading.sets["part3_train"]
    expected = [r for r in rows if 0.5 <= r["split_key"] < 0.8 and r["stars"] != 3]
    assert len(train) == len(expected)
    assert reading.three_stars["part3_train"] == sum(
        1 for r in rows if 0.5 <= r["split_key"] < 0.8 and r["stars"] == 3
    )
    assert train.labels.tolist() == [int(r["stars"] >= 4) for r in expected]
    assert train.words.tolist() == [
        len([t for t in r["text"].split() if t[0].isalpha()]) for r in expected
    ]
    # unknown words become <unk>, as the language models see them
    first = train.reviews.review(0).tolist()
    assert first == vocab.encode(expected[0]["text"].split(" "))

    balanced = balance(train, seed=4213)
    assert (balanced.labels == NEGATIVE).sum() == (balanced.labels == POSITIVE).sum()
    assert (balanced.labels == NEGATIVE).sum() == (train.labels == NEGATIVE).sum()
    assert np.all(np.diff(balanced.reviews.review_ids) > 0)  # corpus order kept
    assert (
        balance(train, seed=4213).reviews.review_ids.tolist()
        == balanced.reviews.review_ids.tolist()
    )
    summary = describe_set(balanced)
    assert summary["negative"] == summary["positive"]
    assert sum(summary["by_length_group"].values()) == len(balanced)

    test = balance(reading.sets["part3_test"], seed=4213)
    copies = copy_checks(test, balanced, reading.lm_train_texts)
    assert copies["share_in_lm_train"] == 1.0  # the five texts recur everywhere
    assert 0.0 <= copies["copy_lookup_accuracy"] <= 1.0


def test_a_product_in_two_piles_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl.gz"
    rows = corpus_rows()
    rows[0]["split_key"] = 0.9  # product p0 now also has a review in another pile
    write_corpus(path, rows)
    vocab = Vocabulary(["<pad>", "<unk>", "<bos>", "<eos>"])
    part3 = {"part3_test": Pile("part3_test", 0.8, 1.0)}
    reading = read_part3(path, part3, {"lm_train": Pile("lm_train", 0.0, 0.5)}, vocab)
    assert reading.shared_products == 1


def test_length_groups_cover_every_review_once() -> None:
    words = np.arange(1, 60)
    groups = ((1, 2), (3, 6), (7, 20), (21, None))
    total = sum(group_mask(words, g).astype(int) for g in groups)
    assert np.all(total == 1)


# --- features --------------------------------------------------------------


@pytest.mark.parametrize(
    "spec",
    [
        {"family": "rnn", "hidden": 12},
        {"family": "lstm", "hidden": 10},
        {"family": "transformer", "width": 16, "layers": 2, "heads": 2, "feedforward": 24},
    ],
)
def test_pooled_features_read_only_the_review_positions(spec: dict) -> None:
    torch.manual_seed(0)
    model = build_model(spec, VOCAB).eval()
    reviews = toy_reviews(25, seed=3)
    batched = pooled_features(model, reviews, 64, torch.device("cpu"), ("pre_output", "hidden"))
    assert batched[("pre_output", "last")].shape == (25, OUTPUT_DIM)
    for i in range(len(reviews)):
        tokens = torch.as_tensor([[2, *reviews.review(i).tolist()]])  # <bos> + the review, no pad
        with torch.no_grad():
            states = model.hidden_states(tokens)[0]
            projected = model.head.projection(states)
        for layer, values in (("hidden", states), ("pre_output", projected)):
            np.testing.assert_allclose(batched[(layer, "last")][i], values[-1].numpy(), atol=1e-5)
            np.testing.assert_allclose(
                batched[(layer, "mean")][i], values[1:].mean(dim=0).numpy(), atol=1e-5
            )


def test_pooling_holds_at_the_longest_review() -> None:
    torch.manual_seed(1)
    spec = {"family": "transformer", "width": 16, "layers": 1, "heads": 2, "feedforward": 16}
    model = build_model(spec, VOCAB).eval()
    rng = np.random.default_rng(7)
    lists = [rng.integers(4, VOCAB, 512).tolist(), [5], rng.integers(4, VOCAB, 37).tolist()]
    reviews = EncodedReviews.from_lists(lists)
    batched = pooled_features(model, reviews, 8192, torch.device("cpu"))
    for i, tokens in enumerate(lists):
        with torch.no_grad():
            states = model.head.projection(model.hidden_states(torch.as_tensor([[2, *tokens]])))[0]
        np.testing.assert_allclose(batched[("pre_output", "last")][i], states[-1], atol=1e-5)
        np.testing.assert_allclose(
            batched[("pre_output", "mean")][i], states[1:].mean(dim=0), atol=1e-5
        )


def test_extraction_changes_no_parameter_and_untrained_copies_repeat() -> None:
    spec = {"family": "lstm", "hidden": 8}
    model = untrained_copy(spec, VOCAB, seed=4213)
    before = parameter_fingerprint(model)
    pooled_features(model, toy_reviews(10, seed=1), 64, torch.device("cpu"))
    assert parameter_fingerprint(model) == before
    torch.manual_seed(4213)
    rebuilt = build_model(spec, VOCAB)
    assert parameter_fingerprint(rebuilt) == before
    assert parameter_fingerprint(untrained_copy(spec, VOCAB, seed=1)) != before


def test_average_vectors_and_counts_match_a_loop() -> None:
    reviews = toy_reviews(20, seed=5)
    table = np.random.default_rng(0).standard_normal((VOCAB, 7)).astype(np.float32)
    averages = average_vectors(reviews, table)
    for i in range(len(reviews)):
        np.testing.assert_allclose(averages[i], table[reviews.review(i)].mean(axis=0), atol=1e-5)
    counts = token_counts(reviews, VOCAB).toarray()
    assert counts.sum() == reviews.ids.size
    documents = ngram_documents(EncodedReviews.from_lists([[5, 6, 7]]), (1, 2))
    assert documents == [["5", "6", "7", "5 6", "6 7"]]
    assert length_feature(np.array([1, 3]))[:, 0].tolist() == pytest.approx(np.log1p([1, 3]))


# --- classifier ------------------------------------------------------------


def separable(n: int, width: int, seed: int, noise: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n)
    features = rng.standard_normal((n, width)) * noise
    features[:, 0] += np.where(labels == 1, 1.0, -1.0)
    return features.astype(np.float32), labels.astype(np.int8)


def test_folds_never_split_a_product_and_keep_the_balance() -> None:
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 300, 3000)
    labels = rng.integers(0, 2, 3000)
    folds = make_folds(labels, groups, 5, seed=4213)
    assert len(folds) == 5
    held = np.concatenate([h for _, h in folds])
    assert np.sort(held).tolist() == list(range(3000))
    for fit, out in folds:
        assert np.intersect1d(groups[fit], groups[out]).size == 0
        assert abs(labels[out].mean() - labels.mean()) < 0.05
    again = make_folds(labels, groups, 5, seed=4213)
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(folds, again, strict=True))


def test_standardising_uses_only_the_fitting_rows() -> None:
    train = np.arange(20, dtype=np.float32).reshape(10, 2)
    features = DenseFeatures(train, train[:3])
    transform = features.fit(np.arange(5))
    fitted = transform(features.rows(np.arange(5)))
    np.testing.assert_allclose(fitted.mean(axis=0), 0.0, atol=1e-6)  # float32 output
    np.testing.assert_allclose(fitted.std(axis=0), 1.0, atol=1e-6)
    later = transform(features.rows(np.arange(5, 10)))
    assert np.all(later > 1.0)  # scaled with the first rows' numbers, not refitted


def test_word_counts_keep_only_ngrams_seen_twice_in_the_fitting_rows() -> None:
    from scipy import sparse

    train = sparse.csr_matrix(
        np.array([[1, 0, 2], [1, 0, 0], [0, 3, 0], [0, 1, 1]], dtype=np.float64)
    )
    features = CountFeatures(train, train, min_documents=2)
    transform = features.fit(np.array([0, 1, 2]))
    assert features.width == 1  # only column 0 occurs in two of rows 0-2
    assert transform(train).shape == (4, 1)


def test_the_c_range_rule() -> None:
    settings = ClassifierSettings(lowest=-8, highest=6)
    assert next_power({-1: 0.80, 0: 0.85, 1: 0.83}, settings) is None  # best inside
    assert next_power({-1: 0.80, 0: 0.83, 1: 0.85}, settings) == 2  # best at the top
    assert next_power({-1: 0.86, 0: 0.83, 1: 0.80}, settings) == -2  # best at the bottom
    assert next_power({-1: 0.85, 0: 0.85, 1: 0.85}, settings) is None  # flat: no pull
    assert next_power({-1: 0.80, 0: 0.85, 1: 0.85}, settings) is None  # tie keeps smaller C
    assert best_power({-1: 0.80, 0: 0.85, 1: 0.85}) == 0
    assert next_power({5: 0.8, 6: 0.9}, settings) is None  # at the limit


def test_the_c_range_grows_when_the_best_is_at_an_end() -> None:
    # x1 = signal + noise and x2 = the same noise: only x1 - x2 separates the classes.
    # Strong shrinkage (small C) leaves the boundary on x1 alone or on the class
    # share and does badly; a weak one learns x1 - x2, so the range must grow upwards.
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 900).astype(np.int8)
    noise = rng.normal(0.0, 3.0, 900)
    features = np.column_stack([np.where(labels == 1, 1.0, -1.0) + noise, noise])
    folds = make_folds(labels, np.arange(900) // 3, 3, seed=0)
    settings = ClassifierSettings(low=-4, high=-2, lowest=-8, highest=3)
    choice = choose_power(DenseFeatures(features, features[:10]), labels, folds, settings)
    assert choice["extended"] >= 1
    assert choice["power"] > -2
    means = choice["mean_accuracy"]
    assert means[str(choice["power"])] == max(means.values())
    assert means["-4"] < 0.7 < means[str(choice["power"])]


def test_fit_and_test_scores_the_test_pile_once() -> None:
    features, labels = separable(800, 4, seed=2)
    test_features, test_labels = separable(300, 4, seed=3)
    folds = make_folds(labels, np.arange(800) // 4, 5, seed=4213)
    result = fit_and_test(
        DenseFeatures(features, test_features), labels, test_labels, folds, ClassifierSettings()
    )
    assert result["correct"].shape == (300,)
    assert result["correct"].mean() > 0.75
    assert result["probability"].min() >= 0.0 and result["probability"].max() <= 1.0
    assert result["width"] == 4


def test_test_labels_never_change_the_choice_or_the_predictions() -> None:
    features, labels = separable(600, 4, seed=8)
    test_features, test_labels = separable(200, 4, seed=9)
    folds = make_folds(labels, np.arange(600) // 3, 5, seed=4213)
    settings = ClassifierSettings()
    first = fit_and_test(
        DenseFeatures(features, test_features), labels, test_labels, folds, settings
    )
    shuffled = np.random.default_rng(0).permutation(test_labels)
    second = fit_and_test(DenseFeatures(features, test_features), labels, shuffled, folds, settings)
    assert first["power"] == second["power"]
    np.testing.assert_array_equal(first["probability"], second["probability"])


def test_tf_idf_weights_come_from_the_fitting_rows_only() -> None:
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfTransformer

    counts = sparse.csr_matrix(np.random.default_rng(3).poisson(0.6, (40, 12)).astype(float))
    features = CountFeatures(counts, counts, min_documents=1)
    rows = np.arange(20)
    transform = features.fit(rows)
    kept = np.flatnonzero(np.asarray((counts[rows] > 0).sum(axis=0)).ravel() >= 1)
    reference = TfidfTransformer(sublinear_tf=True).fit(counts[rows][:, kept])
    np.testing.assert_allclose(
        transform(counts[20:]).toarray(), reference.transform(counts[20:][:, kept]).toarray()
    )


def test_float32_features_give_the_float64_answer() -> None:
    features, labels = separable(2000, 30, seed=10)
    single, _ = fit_logistic(features.astype(np.float32), labels, power=0, max_iter=1000)
    double, _ = fit_logistic(features.astype(np.float64), labels, power=0, max_iter=1000)
    np.testing.assert_allclose(single.coef_, double.coef_, rtol=1e-3, atol=1e-4)
    assert np.array_equal(single.predict(features), double.predict(features))


def test_a_fit_that_stops_early_is_reported_not_fatal() -> None:
    features, labels = separable(400, 5, seed=4)
    _, converged = fit_logistic(features, labels, power=2, max_iter=1)
    assert converged is False
    _, converged = fit_logistic(features, labels, power=0, max_iter=1000)
    assert converged is True


# --- statistics ------------------------------------------------------------


def test_mcnemar_counts_only_the_disagreements() -> None:
    first = np.array([1, 1, 1, 1, 1, 1, 0, 0, 1, 0], dtype=bool)
    second = np.array([1, 0, 0, 0, 0, 1, 1, 0, 1, 0], dtype=bool)
    result = mcnemar(first, second)
    assert (result["first_only"], result["second_only"]) == (4, 1)
    assert result["p"] == pytest.approx(stats.binomtest(1, 5, 0.5).pvalue)
    assert mcnemar(first, first)["p"] == 1.0


def test_holm_matches_a_worked_example() -> None:
    # Sorted: 0.005 x 4 = 0.02, 0.01 x 3 = 0.03, 0.03 x 2 = 0.06, 0.04 x 1 -> 0.06 (kept monotone)
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_pair_agreement_counts_pairs_in_the_same_order() -> None:
    j = np.array([3.6, 3.7, 3.8, 3.9])
    perfect = pair_agreement(j, np.array([0.9, 0.85, 0.8, 0.75]))
    assert perfect["same_order"] == 6 and perfect["share_same_order"] == 1.0
    assert perfect["tau"] == pytest.approx(1.0)
    reversed_ = pair_agreement(j, np.array([0.75, 0.8, 0.85, 0.9]))
    assert reversed_["opposite_order"] == 6 and reversed_["tau"] == pytest.approx(-1.0)
    mixed = pair_agreement(j, np.array([0.9, 0.8, 0.85, 0.75]))
    assert (mixed["same_order"], mixed["opposite_order"]) == (5, 1)


def test_bootstrap_intervals_cover_the_accuracy() -> None:
    rng = np.random.default_rng(0)
    correct = rng.random((2, 2000)) < np.array([[0.8], [0.7]])
    samples = bootstrap_accuracies(correct, 500, seed=1)
    assert samples.shape == (500, 2)
    low, high = interval(samples[:, 0])
    assert low < correct[0].mean() < high
    j = np.array([3.6, 3.9])
    tau_low, tau_high = agreement_interval(j, samples)
    assert tau_low == pytest.approx(1.0) and tau_high == pytest.approx(1.0)
    w_low, w_high = wilson(80, 100)
    assert w_low < 0.8 < w_high


def test_class_scores() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = class_scores(labels, np.array([0, 1, 1, 1]))
    assert scores["accuracy"] == 0.75
    assert scores["accuracy_negative"] == 0.5 and scores["accuracy_positive"] == 1.0
    assert scores["macro_f1"] == pytest.approx(np.mean([2 / 3, 0.8]))


# --- error analysis (D3.4) -------------------------------------------------


def test_negation_and_contrast_words() -> None:
    assert has_negation(["this", "is", "not", "good"])
    assert has_negation(["i", "don't", "like", "it"]) and has_negation(["dont"])
    assert has_negation(["nothing", "special"])
    assert not has_negation(["note", "nobel", "know"])
    assert has_contrast(["good", "but", "pricey"]) and has_contrast(["however", "it"])
    assert not has_contrast(["butter", "buttery"])


def test_groups_split_every_review_and_count_repeats(tmp_path: Path) -> None:
    vocab = Vocabulary(["<pad>", "<unk>", "<bos>", "<eos>", "not", "good", "but", "bad", "."])
    path = tmp_path / "corpus.jsonl.gz"
    texts = ["not good", "good but bad .", "great", "good", "bad bad zzz qqq"]
    rows = [
        {"id": i, "product": f"p{i}", "split_key": 0.9, "stars": [1, 4, 5, 2, 1][i], "text": t}
        for i, t in enumerate(texts)
    ]
    write_corpus(path, rows)
    reading = read_part3(path, {"part3_test": Pile("part3_test", 0.8, 1.0)}, {}, vocab)
    test = reading.sets["part3_test"]
    masks = groups(test, vocab, {hash("good")}, [(1, 2), (3, None)], heavy_unk_share=0.3)
    assert masks["negation"].tolist() == [True, False, False, False, False]
    assert masks["contrast"].tolist() == [False, True, False, False, False]
    assert masks["any <unk>"].tolist() == [False, False, True, False, True]
    assert masks["<unk> share at least 30%"].tolist() == [False, False, True, False, True]
    assert masks["repeats a part3_train review"].tolist() == [False, False, False, True, False]
    assert masks["1 star"].sum() == 2 and masks["all"].all()
    accuracy = group_accuracy(np.array([True, False, True, True, False]), masks)
    assert accuracy["negation"] == {
        "reviews": 1,
        "accuracy": 1.0,
        "interval": accuracy["negation"]["interval"],
    }
    assert accuracy["2 stars"]["reviews"] == 1


def test_disagreement_and_draw() -> None:
    first = np.array([1, 1, 0, 0, 1], dtype=bool)
    second = np.array([1, 0, 1, 0, 1], dtype=bool)
    assert disagreement(first, second) == {
        "both_right": 2,
        "first_only": 1,
        "second_only": 1,
        "both_wrong": 1,
    }
    rng = np.random.default_rng(4213)
    sample = draw(np.arange(100), 10, rng)
    assert sample.size == 10 and np.all(np.diff(sample) > 0)
    assert draw(np.arange(100), 10, np.random.default_rng(4213)).tolist() == sample.tolist()
    assert draw(np.array([3, 7]), 10, rng).tolist() == [3, 7]
