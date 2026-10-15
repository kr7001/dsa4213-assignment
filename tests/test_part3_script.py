"""End-to-end run of scripts/part3.py on a tiny synthetic corpus and untrained models."""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from a1.data.sequences import encode_piles, load_piles
from a1.data.vocab import Vocabulary
from a1.evaluation.perplexity import evaluate
from a1.models.build import build_model, count_parameters
from a1.models.pretrained import install_table, table_sha256

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "part3.py"
PILES = {
    "lm_train": [0.0, 0.5],
    "lm_valid": [0.5, 0.55],
    "lm_test": [0.55, 0.6],
    "part3_test": [0.6, 0.7],
    "part3_train": [0.7, 1.0],
}
SPECS = {
    "rnn-S-full": {"family": "rnn", "hidden": 12},
    "lstm-S-full": {"family": "lstm", "hidden": 10},
    "transformer-S-full": {
        "family": "transformer",
        "width": 100,
        "layers": 1,
        "heads": 4,
        "feedforward": 16,
    },
    "lstm-own-frozen-s4213-full": {
        "family": "lstm",
        "hidden": 10,
        "embedding": "data/embeddings/skipgram-scaled.npy",
        "freeze": True,
        "seed": 4213,
    },
    "lstm-random-s4214-full": {"family": "lstm", "hidden": 10, "freeze": False, "seed": 4214},
}


def write_world(root: Path) -> Path:
    rng = np.random.default_rng(0)
    good = [f"g{i}" for i in range(15)]
    bad = [f"b{i}" for i in range(15)]
    plain = [f"w{i}" for i in range(30)]
    rows = []
    for i in range(3000):
        product = int(rng.integers(0, 300))
        stars = int(rng.integers(1, 6))
        pool = good if stars >= 4 else bad if stars <= 2 else plain
        n = int(rng.integers(1, 25))
        words = [
            pool[int(rng.integers(0, len(pool)))]
            if rng.random() < 0.3
            else plain[int(rng.integers(0, 30))]
            for _ in range(n)
        ]
        rows.append(
            {
                "id": i,
                "product": f"p{product}",
                "split_key": product / 300,
                "stars": stars,
                "text": " ".join(words),
            }
        )
    corpus = root / "data" / "processed" / "reviews.jsonl.gz"
    corpus.parent.mkdir(parents=True)
    with gzip.open(corpus, "wt", encoding="utf-8") as handle:
        handle.writelines(json.dumps(row) + "\n" for row in rows)
    piles_path = root / "piles.yaml"
    piles_path.write_text(yaml.safe_dump({"piles": PILES}))
    piles = load_piles(piles_path)
    counts: Counter = Counter()
    for row in rows:
        if piles["lm_train"].contains(row["split_key"]):
            counts.update(row["text"].split(" "))
    vocab = Vocabulary.build(counts, min_count=2)
    vocab.save(root / "data" / "processed" / "vocab.json")
    valid = encode_piles(corpus, {"lm_valid": piles["lm_valid"]}, vocab)["lm_valid"]

    tables = root / "data" / "embeddings"
    tables.mkdir(parents=True)
    for name in ("skipgram-scaled", "glove"):
        table = np.random.default_rng(len(name)).standard_normal((len(vocab), 100)) * 0.5
        table[0] = 0.0
        np.save(tables / f"{name}.npy", table.astype(np.float32))

    runs, checkpoints = root / "results" / "runs", root / "checkpoints"
    checkpoints.mkdir()
    test_scores = {}
    for name, spec in SPECS.items():
        seed = int(spec.get("seed", 4213))
        torch.manual_seed(seed)
        model = build_model(spec, len(vocab))
        table = np.load(root / spec["embedding"]) if spec.get("embedding") else None
        install_table(model, table, bool(spec.get("freeze", False)))
        start = table_sha256(model.embedding.weight)
        with torch.no_grad():  # stand-in for training: the saved weights differ from the start
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.add_(0.05 * torch.randn_like(parameter))
        j = evaluate(model, valid, 512, torch.device("cpu"))["J"]
        record = {
            "run": name,
            "spec": spec,
            "parameters": count_parameters(model),
            "summary": {
                "best_valid_J": j,
                "train_seconds": 60.0,
                "config": {"seed": seed, "peak_lr": 0.01},
            },
        }
        if "seed" in spec:  # Part II records keep the seed and the table fingerprints
            record["seed"] = seed
            record["embedding"] = {
                "table": spec.get("embedding", "random start"),
                "start_sha256": start,
                "end_sha256": table_sha256(model.embedding.weight),
            }
        (runs / name).mkdir(parents=True)
        (runs / name / "run.json").write_text(json.dumps(record))
        torch.save(model.state_dict(), checkpoints / f"{name}.pt")
        test_scores[name] = {"test": {"J": j + 0.01}}
    (root / "results" / "eval").mkdir(parents=True)
    (root / "results" / "eval" / "test.json").write_text(json.dumps({"models": test_scores}))

    expected = {}
    for pile in ("part3_train", "part3_test"):
        stars = [r["stars"] for r in rows if piles[pile].contains(r["split_key"])]
        negative = sum(1 for s in stars if s <= 2)
        positive = sum(1 for s in stars if s >= 4)
        expected[pile] = 2 * min(negative, positive)

    config = {
        "corpus": str(corpus),
        "piles": str(piles_path),
        "vocabulary": str(root / "data" / "processed" / "vocab.json"),
        "runs": str(runs),
        "checkpoints": str(checkpoints),
        "test_scores": str(root / "results" / "eval" / "test.json"),
        "results": str(root / "results" / "part3"),
        "cache": str(root / "data" / "part3"),
        "device": "cpu",
        "batch_tokens": 512,
        "seed": 4213,
        "expected_reviews": expected,
        "models": {
            "design": ["rnn-S-full", "lstm-S-full", "transformer-S-full"],
            "word_table": ["lstm-own-frozen-s4213-full"],
            "seed_repeats": ["lstm-random-s4214-full"],
        },
        "ranking": [
            "rnn-S-full",
            "lstm-S-full",
            "transformer-S-full",
            "lstm-own-frozen-s4213-full",
        ],
        "ranking_groups": {"design": ["rnn-S-full", "lstm-S-full", "transformer-S-full"]},
        "seed_groups": {"random_trainable": ["lstm-S-full", "lstm-random-s4214-full"]},
        "layer": "pre_output",
        "poolings": ["last", "mean"],
        "hidden_check": ["rnn-S-full"],
        "untrained": ["lstm-S-full"],
        "tables": {
            "learned": {"checkpoint": "lstm-S-full"},
            "skipgram": {"file": str(tables / "skipgram-scaled.npy")},
            "glove": {"file": str(tables / "glove.npy")},
        },
        "word_counts": {"orders": [1, 2], "min_documents": 2, "sublinear_tf": True},
        "classifier": {"folds": 3, "powers": [-2, 2], "limits": [-4, 4], "max_iter": 2000},
        "bootstrap": 50,
        "length_groups": [[1, 2], [3, 6], [7, 20], [21, None]],
        "minimum_words": [3, 5],
        "comparisons": [
            ["lstm-S-full", "untrained/lstm-S-full"],
            ["rnn-S-full", "lstm-S-full"],
            ["lstm-S-full", "word_counts"],
        ],
    }
    path = root / "part3.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def load_script(monkeypatch: pytest.MonkeyPatch):
    spec = importlib.util.spec_from_file_location("part3_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)  # dataclasses look their module up
    spec.loader.exec_module(module)
    return module


def test_part3_script_runs_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = write_world(tmp_path)
    monkeypatch.chdir(tmp_path)
    module = load_script(monkeypatch)
    assert module.main(["--config", str(config_path)]) == 0

    result = json.loads((tmp_path / "results" / "part3" / "part3.json").read_text())
    reps = result["representations"]
    expected = {
        *(f"{m}/pre_output/{p}" for m in SPECS for p in ("last", "mean")),
        "rnn-S-full/hidden/last",
        "rnn-S-full/hidden/mean",
        "untrained/lstm-S-full/pre_output/last",
        "untrained/lstm-S-full/pre_output/mean",
        "table/learned",
        "table/skipgram",
        "table/glove",
        "word_counts",
        "length",
    }
    assert set(reps) == expected
    assert reps["lstm-S-full/pre_output/last"]["width"] == 100
    assert reps["rnn-S-full/hidden/last"]["width"] == 12
    assert result["data"]["products_shared_with_other_piles"] == 0
    assert all(fold["products_on_both_sides"] == 0 for fold in result["data"]["folds"])
    assert result["checks"]["untrained_start"]["same_start"] is True
    # 3 pairs at 2 poolings + last vs mean for 4 ranked models
    assert len(result["comparisons"]) == 3 * 2 + 4
    assert all(row["p_holm"] >= row["p"] for row in result["comparisons"])
    overall = result["ranking"]["last"]["overall"]
    assert overall["pairs"] == 6
    assert overall["same_order"] + overall["opposite_order"] + overall["ties"] == 6
    assert set(result["luck"]["mean"]["settings"]) == {"random_trainable"}
    assert all(
        info["parameters_unchanged"]
        for name, layers in result["sources"].items()
        if name in SPECS
        for info in layers.values()
    )
    # the planted sentiment words must be found, so features and labels line up
    for name in ("word_counts", "table/learned", "transformer-S-full/pre_output/mean"):
        assert reps[name]["test"]["accuracy"] > 0.8, name
    low, high = reps["word_counts"]["test"]["interval_by_product"]
    assert low <= reps["word_counts"]["test"]["accuracy"] <= high

    with np.load(tmp_path / "data" / "part3" / "predictions.npz") as stored:
        names = stored["names"].tolist()
        assert stored["probability"].shape == (len(expected), len(stored["labels"]))
        trained = stored["probability"][names.index("lstm-S-full/pre_output/mean")]
        untrained = stored["probability"][names.index("untrained/lstm-S-full/pre_output/mean")]
        assert not np.allclose(trained, untrained)  # the untrained copy is not the checkpoint
        correct = stored["predicted"] == stored["labels"][None, :]
        assert correct[names.index("word_counts")].mean() == reps["word_counts"]["test"]["accuracy"]
        assert stored["products"].shape == stored["labels"].shape

    # a second run reuses every finished classifier and gives the same numbers
    capsys.readouterr()
    assert module.main(["--config", str(config_path)]) == 0
    log = capsys.readouterr().out
    assert log.count("done in an earlier run; reused") == len(expected)
    again = json.loads((tmp_path / "results" / "part3" / "part3.json").read_text())
    for name in expected:
        assert again["representations"][name]["test"] == reps[name]["test"], name
    assert again["costs"]["lstm-S-full"]["feature_seconds_test_pile"] is not None

    # a half-written file (a run stopped mid-write) is redone, not fatal
    classified = tmp_path / "data" / "part3" / "classified" / "lstm-S-full__pre_output__mean.npz"
    classified.write_bytes(classified.read_bytes()[:100])
    features = tmp_path / "data" / "part3" / "features" / "lstm-S-full.npz"
    features.write_bytes(features.read_bytes()[:100])
    capsys.readouterr()
    assert module.main(["--config", str(config_path)]) == 0
    log = capsys.readouterr().out
    assert log.count("done in an earlier run; reused") == len(expected) - 1
    third = json.loads((tmp_path / "results" / "part3" / "part3.json").read_text())
    name = "lstm-S-full/pre_output/mean"
    assert third["representations"][name]["test"] == reps[name]["test"]


def test_the_hidden_check_can_be_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = write_world(tmp_path)
    monkeypatch.chdir(tmp_path)
    module = load_script(monkeypatch)
    assert module.main(["--config", str(config_path), "--without-hidden-check"]) == 0
    result = json.loads((tmp_path / "results" / "part3" / "part3.json").read_text())
    assert result["hidden_check_included"] is False
    assert not any("/hidden/" in name for name in result["representations"])


def test_a_wrong_checkpoint_stops_before_any_feature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = write_world(tmp_path)
    record_path = tmp_path / "results" / "runs" / "rnn-S-full" / "run.json"
    record = json.loads(record_path.read_text())
    record["summary"]["best_valid_J"] += 0.01
    record_path.write_text(json.dumps(record))
    monkeypatch.chdir(tmp_path)
    module = load_script(monkeypatch)
    with pytest.raises(SystemExit, match="rnn-S-full"):
        module.main(["--config", str(config_path)])
    assert not (tmp_path / "results" / "part3").exists()
    assert not (tmp_path / "data" / "part3").exists()
