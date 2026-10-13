"""Tests for Part II: building the input tables and keeping a fixed table fixed."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from a1.data.sequences import EncodedReviews
from a1.data.vocab import BOS_ID, PAD_ID, SPECIAL_TOKENS, Vocabulary
from a1.embedding_tables import (
    check_skipgram_settings,
    coverage,
    glove_table,
    glove_wanted,
    informative_rows,
    input_counts,
    intrinsic_scores,
    nearest_neighbours,
    random_table,
    read_glove,
    review_sentences,
    rms,
    scaled_to,
    split_assimilation,
    table_stats,
    train_skipgram,
)
from a1.models.build import build_model, count_parameters, describe
from a1.models.pretrained import (
    embedding_change,
    install_table,
    load_table,
    table_sha256,
)
from a1.training.trainer import TrainConfig, make_optimizer, train

CPU = torch.device("cpu")
WORDS = ["the", "great", "don't", "cannot", "gonna", "<num>", "zzqx", "it's", "do", "cat"]


def _vocab(words: list[str] = WORDS) -> Vocabulary:
    return Vocabulary([*SPECIAL_TOKENS, *words])


def _toy_reviews(n: int, seed: int, vocab_size: int = 30) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    reviews = []
    for _ in range(n):
        start, length = rng.integers(4, vocab_size), rng.integers(3, 12)
        reviews.append([4 + (start - 4 + i) % (vocab_size - 4) for i in range(length)])
    return reviews


def test_assimilations_split_like_corenlp() -> None:
    assert split_assimilation("cannot") == ["can", "not"]
    assert split_assimilation("gonna") == ["gon", "na"]
    assert split_assimilation("dont") == ["do", "nt"]
    assert split_assimilation("theyre") == ["they", "re"]
    assert split_assimilation("dunno") == ["du", "n", "no"]  # "nno" is split again
    assert split_assimilation("wont") == ["wont"]  # a real word: CoreNLP leaves it alone
    assert split_assimilation("hello") == ["hello"]


def test_glove_rows_come_from_the_right_source() -> None:
    vocab = _vocab()
    rng = np.random.default_rng(0)
    vectors = {
        w: rng.normal(size=4).astype(np.float32)
        for w in ["the", "great", "do", "n't", "it", "'s", "can", "not", "gon", "na", "cat"]
    }
    numbers = ["1", "2"]
    vectors |= {"1": np.ones(4, np.float32), "2": 3 * np.ones(4, np.float32)}
    vectors["<unk>"] = np.full(4, 9.0, np.float32)  # GloVe's own <unk> line is not used
    table, sources = glove_table(vocab, vectors, numbers, dim=4, seed=4213)
    source = dict(zip(vocab.tokens, sources, strict=True))
    assert source["<pad>"] == "pad" and not table[PAD_ID].any()
    assert source["the"] == "exact" and np.array_equal(table[vocab.index["the"]], vectors["the"])
    assert source["don't"] == "contraction"
    assert np.allclose(table[vocab.index["don't"]], (vectors["do"] + vectors["n't"]) / 2)
    assert source["it's"] == "contraction"
    assert source["cannot"] == "assimilation" and source["gonna"] == "assimilation"
    assert np.allclose(table[vocab.index["cannot"]], (vectors["can"] + vectors["not"]) / 2)
    assert source["<num>"] == "number" and np.allclose(table[vocab.index["<num>"]], 2.0)
    for token in ("<unk>", "<bos>", "<eos>", "zzqx"):
        assert source[token] == "random"
    assert not np.allclose(table[vocab.index["<unk>"]], 9.0)
    # Random rows: uniform in [-a, a] with the mean square of the exact rows (Kim, 2014).
    exact = np.array([s == "exact" for s in sources])
    bound = math.sqrt(3 * np.mean(table[exact].astype(np.float64) ** 2))
    random_rows = table[[s == "random" for s in sources]]
    assert np.all(np.abs(random_rows) <= bound + 1e-6) and np.abs(random_rows).max() > 0
    again, _ = glove_table(vocab, vectors, numbers, dim=4, seed=4213)
    assert np.array_equal(table, again)  # the same seed gives the same random rows


def test_random_rows_match_the_mean_square_of_the_glove_rows() -> None:
    known = [f"k{i}" for i in range(60)]
    missing = [f"m{i}" for i in range(300)]
    vocab = _vocab(known + missing)
    rng = np.random.default_rng(5)
    vectors = {w: rng.normal(scale=0.4, size=100).astype(np.float32) for w in known}
    table, sources = glove_table(vocab, vectors, [], dim=100, seed=4213)
    exact = np.array([s == "exact" for s in sources])
    random_rows = np.array([s == "random" for s in sources])
    exact_ms = np.mean(table[exact].astype(np.float64) ** 2)
    random_ms = np.mean(table[random_rows].astype(np.float64) ** 2)
    assert random_ms == pytest.approx(exact_ms, rel=0.03)  # Kim (2014): the same variance
    # The fixed random table draws from its own stream: no row repeats a GloVe random row
    # (with one shared stream, rows would be exact multiples, correlation 1).
    other = random_table(len(vocab), 100, 0.4, seed=4213)[1:]
    a = table[random_rows] - table[random_rows].mean(axis=1, keepdims=True)
    b = other - other.mean(axis=1, keepdims=True)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    assert np.abs(a @ b.T).max() < 0.9


def test_read_glove_keeps_wanted_words_and_frequent_numbers(tmp_path: Path) -> None:
    lines = ["5 4"]  # a gensim-style header, skipped
    for i, word in enumerate(["the", ",", "2019", "cat", "12.5", "dog", "the", "7"]):
        lines.append(" ".join([word] + [f"{i + 0.5 * k:.2f}" for k in range(4)]))
    path = tmp_path / "glove.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("readme.txt", "small")
        archive.writestr("vectors.txt", "\n".join(lines) + "\n")
    vectors, numbers, facts = read_glove(path, {"the", "cat", "fish"}, dim=4, number_top=6)
    assert set(vectors) == {"the", "cat", "2019", "12.5"}  # "7" is not among the first six
    assert numbers == ["2019", "12.5"]
    assert np.allclose(vectors["the"], [0.0, 0.5, 1.0, 1.5])  # a repeated word keeps its first
    assert facts["words"] == 8 and facts["first_words"][:3] == ["the", ",", "2019"]
    with pytest.raises(ValueError, match="numbers"):
        read_glove(path, {"cat"}, dim=5)


def test_wanted_words_include_the_pieces() -> None:
    wanted = glove_wanted(_vocab())
    assert {"do", "n't", "it", "'s", "can", "not", "gon", "na", "the"} <= wanted


def test_scaling_keeps_every_cosine_and_reaches_the_target() -> None:
    rng = np.random.default_rng(1)
    table = rng.normal(size=(50, 8)).astype(np.float32)
    table[PAD_ID] = 0
    scaled, factor = scaled_to(table, 0.25)
    assert rms(scaled) == pytest.approx(0.25, rel=1e-5) and factor > 0
    unit = table[1:] / np.linalg.norm(table[1:], axis=1, keepdims=True)
    unit_scaled = scaled[1:] / np.linalg.norm(scaled[1:], axis=1, keepdims=True)
    assert np.allclose(unit @ unit.T, unit_scaled @ unit_scaled.T, atol=1e-5)


def test_random_table_has_the_target_scale_and_a_zero_pad_row() -> None:
    table = random_table(2000, 100, 0.4, seed=4213)
    assert not table[PAD_ID].any()
    assert rms(table) == pytest.approx(0.4, rel=0.01)
    assert np.abs(table).max() <= math.sqrt(3) * 0.4 + 1e-6


def test_norm_check_and_coverage_add_up() -> None:
    vocab = _vocab()
    table = np.zeros((len(vocab), 2), np.float32)
    table[1:] = [3.0, 4.0]  # every row has length 5
    counts = np.arange(len(vocab))
    stats = table_stats(table, counts)
    assert stats["row_norm_median"] == pytest.approx(5.0)
    assert stats["row_norm_input_weighted"] == pytest.approx(5.0)
    assert stats["rms"] == pytest.approx(math.sqrt(12.5))
    sources = ["pad", *["random"] * 3, *["exact"] * (len(vocab) - 4)]
    report = coverage(sources, vocab, counts)
    shares = report["by_source"]
    assert sum(s["types"] for s in shares.values()) == len(vocab)
    assert sum(s["input_share"] for s in shares.values()) == pytest.approx(1.0)
    assert list(report["most_frequent_random"]) == ["<eos>", "<bos>", "<unk>"]
    assert report["word_types"] == len(vocab) - 4 == report["word_types_with_glove_vector"]
    assert report["input_share_with_glove_vector"] == pytest.approx(counts[4:].sum() / counts.sum())


def test_input_counts_add_one_bos_per_review() -> None:
    reviews = EncodedReviews.from_lists([[5, 6], [5]])
    counts = input_counts(reviews, 8)
    assert counts[BOS_ID] == 2 and counts[5] == 2 and counts[6] == 1 and counts.sum() == 5


def test_skipgram_gives_every_token_a_vector() -> None:
    vocab = Vocabulary([*SPECIAL_TOKENS, *[f"w{i}" for i in range(26)]])
    reviews = EncodedReviews.from_lists(_toy_reviews(200, 0))
    sentences = review_sentences(reviews, vocab)
    assert sentences[0][0] == "<bos>" and sentences[0][-1] == "<eos>"
    table, record = train_skipgram(sentences, vocab, {"vector_size": 16, "epochs": 2})
    assert table.shape == (len(vocab), 16) and not table[PAD_ID].any()
    assert record["vocabulary_tokens_without_vector"] == ["<unk>"]  # never in this text
    assert record["tokens_per_epoch"] == sum(len(s) for s in sentences)
    assert record["seconds"] > 0 and record["settings"]["sg"] == 1


def test_skipgram_settings_are_checked() -> None:
    assert check_skipgram_settings({})["window"] == 5
    with pytest.raises(ValueError, match="workers"):
        check_skipgram_settings({"workers": 4})
    with pytest.raises(ValueError, match="Unknown"):
        check_skipgram_settings({"size": 100})


def test_skipgram_is_the_same_in_every_python_process() -> None:
    code = (
        "import hashlib, numpy as np\n"
        "from a1.data.sequences import EncodedReviews\n"
        "from a1.data.vocab import SPECIAL_TOKENS, Vocabulary\n"
        "from a1.embedding_tables import review_sentences, train_skipgram\n"
        "vocab = Vocabulary([*SPECIAL_TOKENS, *[f'w{i}' for i in range(26)]])\n"
        "rng = np.random.default_rng(0)\n"
        "reviews = EncodedReviews.from_lists("
        "[list(rng.integers(4, 30, size=8)) for _ in range(200)])\n"
        "table, _ = train_skipgram(review_sentences(reviews, vocab), vocab, "
        "{'vector_size': 16, 'epochs': 2})\n"
        "print(hashlib.sha256(table.tobytes()).hexdigest())\n"
    )
    digests = {
        subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for hash_seed in ("1", "2")
    }
    assert len(digests) == 1


def test_intrinsic_checks_use_the_same_words_for_every_table() -> None:
    words = ["tiger", "cat", "love", "sex", "computer", "keyboard", "king", "queen", "man"]
    vocab = _vocab(words)
    rng = np.random.default_rng(2)
    tables = {
        "a": rng.normal(size=(len(vocab), 8)).astype(np.float32),
        "b": rng.normal(size=(len(vocab), 8)).astype(np.float32),
    }
    rows = informative_rows(["pad", "random", "random", "random", *["exact"] * len(words)], vocab)
    assert not rows[:4].any() and rows[4:].all()
    scores = intrinsic_scores(tables, vocab, rows)
    assert scores["words"] == len(words)
    assert scores["a"]["wordsim353_pairs"] == scores["b"]["wordsim353_pairs"] > 0
    assert scores["a"]["wordsim353_pairs_total"] == 353
    assert scores["a"]["analogy_questions"] == scores["b"]["analogy_questions"]
    small = {"a": np.ones((6, 8), np.float32)}
    nothing = intrinsic_scores(small, _vocab(["zzqx", "qqzx"]), np.array([False] * 4 + [True] * 2))
    assert nothing["a"]["wordsim353_spearman"] is None and nothing["a"]["analogy_accuracy"] is None
    neighbours = nearest_neighbours(tables, vocab, rows, queries=("tiger", "absent"), k=3)
    assert set(neighbours["a"]) == {"tiger"} and len(neighbours["a"]["tiger"]) == 3
    assert "tiger" not in neighbours["a"]["tiger"]


def test_nearest_neighbours_are_ranked_by_cosine() -> None:
    vocab = _vocab(["a", "b", "c", "d"])
    table = np.zeros((len(vocab), 2), np.float32)
    table[4:] = [[1, 0], [0.9, 0.1], [0, 1], [-1, 0]]
    rows = np.array([False] * 4 + [True] * 4)
    found = nearest_neighbours({"t": table}, vocab, rows, queries=("a",), k=3)
    assert found["t"]["a"] == ["b", "c", "d"]


def test_a_frozen_table_stays_bit_identical_through_training(tmp_path: Path) -> None:
    vocab_size = 30
    rng = np.random.default_rng(3)
    table = rng.normal(scale=0.4, size=(vocab_size, 100)).astype(np.float32)
    table[PAD_ID] = 0
    np.save(tmp_path / "t.npy", table)
    loaded = load_table(tmp_path / "t.npy", vocab_size, 100)
    torch.manual_seed(4213)
    model = build_model({"family": "lstm", "hidden": 16}, vocab_size)
    install_table(model, loaded, freeze=True)
    weight = model.embedding.weight
    assert not weight.requires_grad and table_sha256(weight) == table_sha256(table)
    optimizer = make_optimizer(model, TrainConfig())
    in_optimizer = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert id(weight) not in in_optimizer
    counts = count_parameters(model)
    assert counts["trainable"] == counts["total"] - vocab_size * 100
    before = weight.detach().clone()
    train_data = EncodedReviews.from_lists(_toy_reviews(200, 4))
    valid_data = EncodedReviews.from_lists(_toy_reviews(30, 5))
    config = TrainConfig(peak_lr=1e-2, batch_tokens=256, epochs=1, checks_per_epoch=2)
    summary = train(model, train_data, valid_data, config, CPU, log=lambda _: None)
    change = embedding_change(before, model.embedding.weight)
    assert change["unchanged"] and change["relative_change"] == 0.0
    assert "embedding_gradient_norms" not in summary  # E has no gradient when fixed


def test_a_trainable_table_moves_and_its_gradient_is_measured() -> None:
    vocab_size = 30
    table = np.random.default_rng(6).normal(scale=0.4, size=(vocab_size, 100)).astype(np.float32)
    table[PAD_ID] = 0
    torch.manual_seed(4213)
    model = build_model({"family": "lstm", "hidden": 16}, vocab_size)
    install_table(model, table, freeze=False)
    before = model.embedding.weight.detach().clone()
    train_data = EncodedReviews.from_lists(_toy_reviews(200, 7))
    config = TrainConfig(peak_lr=1e-2, batch_tokens=256, epochs=1)
    summary = train(model, train_data, None, config, CPU, log=lambda _: None)
    change = embedding_change(before, model.embedding.weight)
    assert not change["unchanged"] and change["relative_change"] > 0
    assert not model.embedding.weight[PAD_ID].any()  # <pad> never receives a gradient
    total = summary["gradient_norms"]
    without = summary["gradient_norms_without_embedding"]
    e_norms = summary["embedding_gradient_norms"]
    assert e_norms["steps"] == without["steps"] == total["steps"] == summary["steps"]
    assert without["mean"] <= total["mean"] and e_norms["mean"] > 0


def test_every_setting_starts_the_lstm_and_head_from_the_same_numbers() -> None:
    def start(table: np.ndarray | None, freeze: bool) -> dict[str, torch.Tensor]:
        torch.manual_seed(4214)
        model = build_model({"family": "lstm", "hidden": 16}, 30)
        install_table(model, table, freeze)
        return model.state_dict()

    table = np.full((30, 100), 0.3, np.float32)
    table[PAD_ID] = 0
    random_start, fixed_glove = start(None, False), start(table, True)
    for name, value in random_start.items():
        if name != "embedding.weight":
            assert torch.equal(value, fixed_glove[name]), name
    assert not torch.equal(random_start["embedding.weight"], fixed_glove["embedding.weight"])


def test_loading_rejects_bad_tables(tmp_path: Path) -> None:
    good = np.zeros((10, 4), np.float32)
    np.save(tmp_path / "shape.npy", good)
    with pytest.raises(ValueError, match="shape"):
        load_table(tmp_path / "shape.npy", 11, 4)
    bad_pad = good.copy()
    bad_pad[PAD_ID] = 1.0
    np.save(tmp_path / "pad.npy", bad_pad)
    with pytest.raises(ValueError, match="pad"):
        load_table(tmp_path / "pad.npy", 10, 4)
    not_finite = good.copy()
    not_finite[3, 1] = np.nan
    np.save(tmp_path / "nan.npy", not_finite)
    with pytest.raises(ValueError, match="finite"):
        load_table(tmp_path / "nan.npy", 10, 4)


def test_part2_labels_show_the_table_and_the_seed() -> None:
    assert describe({"family": "lstm", "hidden": 248}) == "lstm H=248 x1"
    spec = {"family": "lstm", "hidden": 248, "embedding": "data/embeddings/glove.npy"}
    assert describe({**spec, "freeze": True, "seed": 4214}) == (
        "lstm H=248 x1, E glove.npy frozen, seed 4214"
    )
    assert describe({"family": "lstm", "hidden": 248, "freeze": False}) == (
        "lstm H=248 x1, E random trainable"
    )


def test_tuning_can_start_from_another_grid() -> None:
    from a1.training.tuning import next_learning_rate

    start = (0.003, 0.01, 0.03)
    tries: dict[float, float] = {}
    order = []
    j = {0.003: 3.85, 0.01: 3.71, 0.03: 3.70, 0.1: 3.9}
    while (rate := next_learning_rate(tries, start)) is not None:
        order.append(rate)
        tries[rate] = j[rate]
    assert order == [0.003, 0.01, 0.03, 0.1]
    with pytest.raises(ValueError, match="ladder"):
        next_learning_rate({}, (0.005,))


def test_a_finished_run_is_only_reused_with_the_same_settings() -> None:
    import importlib.util

    spec_file = importlib.util.spec_from_file_location(
        "train_lm", Path(__file__).resolve().parents[1] / "scripts" / "train_lm.py"
    )
    train_lm = importlib.util.module_from_spec(spec_file)
    spec_file.loader.exec_module(train_lm)
    spec = {"family": "lstm", "hidden": 248, "embedding": "t.npy", "freeze": True, "seed": 4214}
    record = {
        "spec": dict(spec),
        "summary": {"config": {"peak_lr": 0.01, "epochs": 6.0, "seed": 4214}},
        "embedding": {"file_sha256": "abc"},
    }
    assert train_lm.settings_differ(record, spec, 0.01, 6, 4214, "abc") == []
    assert train_lm.settings_differ(record, spec, 0.03, 6, 4214, "abc")  # another rate
    assert train_lm.settings_differ(record, spec, 0.01, 10, 4214, "abc")  # another length
    assert train_lm.settings_differ(record, spec, 0.01, 6, 4214, "rebuilt")  # another table
    assert train_lm.settings_differ(record, {**spec, "freeze": False}, 0.01, 6, 4214, "abc")
    part1 = {"spec": {"family": "lstm", "hidden": 248}, "summary": record["summary"]}
    assert (
        train_lm.settings_differ(part1, {"family": "lstm", "hidden": 248}, 0.01, 6, 4214, None)
        == []
    )
