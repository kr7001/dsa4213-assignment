"""End-to-end run of scripts/evaluate.py on a tiny synthetic corpus and untrained models."""

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
from a1.models.ngram import AddK, KneserNey, NgramCounts
from a1.models.pretrained import table_sha256

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
SPECS = {
    "rnn-S-full": {"family": "rnn", "hidden": 12},
    "lstm-S-full": {"family": "lstm", "hidden": 12},
    "transformer-S-full": {
        "family": "transformer",
        "width": 100,
        "layers": 1,
        "heads": 4,
        "feedforward": 16,
    },
    "lstm-own-frozen-s4213-full": {
        "family": "lstm",
        "hidden": 12,
        "embedding": "data/embeddings/skipgram-scaled.npy",
        "freeze": True,
        "seed": 4213,
    },
}


def write_world(root: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    words = [f"w{i}" for i in range(40)]
    rows = []
    for i in range(1200):
        n = int(rng.integers(4, 41))
        picks = np.minimum(rng.geometric(0.12, size=n) - 1, len(words) - 1)
        text = " ".join(words[j] for j in picks)
        rows.append({"id": i, "split_key": float(rng.uniform(0, 0.6)), "text": text})
    corpus = root / "data" / "reviews.jsonl.gz"
    corpus.parent.mkdir(parents=True)
    with gzip.open(corpus, "wt", encoding="utf-8") as handle:
        handle.writelines(json.dumps(row) + "\n" for row in rows)
    piles_path = root / "piles.yaml"
    piles_path.write_text(
        yaml.safe_dump(
            {"piles": {"lm_train": [0.0, 0.5], "lm_valid": [0.5, 0.55], "lm_test": [0.55, 0.6]}}
        )
    )
    piles = load_piles(piles_path)
    counts: Counter = Counter()
    for row in rows:
        if piles["lm_train"].contains(row["split_key"]):
            counts.update(row["text"].split(" "))
    vocab = Vocabulary.build(counts)
    vocab.save(root / "data" / "vocab.json")
    data = encode_piles(corpus, {k: piles[k] for k in ("lm_train", "lm_valid")}, vocab)

    table = NgramCounts.build(data["lm_train"], len(vocab), 5)
    kn_valid = float(-KneserNey(table, 5).log_probs(data["lm_valid"]).mean())
    addk = AddK(table, 5, 0.01)
    addk.fit_weights(data["lm_valid"])
    addk_valid = float(-addk.log_probs(data["lm_valid"]).mean())
    ngram = {
        "validation": [
            {"method": "kneser_ney", "order": 5, "valid": {"J": kn_valid}},
            {"method": "add_k", "order": 5, "k": 0.01, "valid": {"J": addk_valid}},
        ],
        "chosen": {"kneser_ney": {"order": 5, "k": None}, "add_k": {"order": 5, "k": 0.01}},
        "counting_seconds": 0.1,
        "peak_memory_bytes": 1,
    }
    (root / "results" / "ngram").mkdir(parents=True)
    (root / "results" / "ngram" / "ngram.json").write_text(json.dumps(ngram))

    runs, checkpoints = root / "results" / "runs", root / "checkpoints"
    checkpoints.mkdir()
    for index, (name, spec) in enumerate(
        [*SPECS.items(), ("rnn-S-try-lr0.01", SPECS["rnn-S-full"])]
    ):
        torch.manual_seed(index)
        model = build_model(spec, len(vocab))
        best = evaluate(model, data["lm_valid"], 512, torch.device("cpu"))["J"]
        record = {
            "run": name,
            "model": name.split("-full")[0],
            "spec": spec,
            "parameters": count_parameters(model),
            "summary": {
                "best_valid_J": best,
                "config": {"peak_lr": 0.01, "epochs": 1, "batch_tokens": 512},
                "seconds": 2.0,
                "train_seconds": 1.5,
                "eval_seconds": 0.5,
                "seconds_to_best": 2.0,
                "train_seconds_to_best": 1.5,
                "steps": 10,
                "best_step": 10,
                "epochs_done": 1.0,
                "targets_seen": 1000,
                "stopped_early": False,
            },
        }
        if spec.get("freeze"):
            record["embedding"] = {"end_sha256": table_sha256(model.embedding.weight.detach())}
        (runs / name).mkdir(parents=True)
        (runs / name / "run.json").write_text(json.dumps(record))
        torch.save(model.state_dict(), checkpoints / f"{name}.pt")

    eval_config = {
        "corpus": str(corpus),
        "piles": str(piles_path),
        "vocabulary": str(root / "data" / "vocab.json"),
        "ngram_results": str(root / "results" / "ngram" / "ngram.json"),
        "embeddings_results": str(root / "missing.json"),
        "runs": str(runs),
        "checkpoints": str(checkpoints),
        "results": str(root / "results" / "eval"),
        "device": "cpu",
        "batch_tokens": 512,
        "models": {
            "part1_main": ["rnn-S-full", "lstm-S-full", "transformer-S-full"],
            "part2": ["lstm-S-full", "lstm-own-frozen-s4213-full"],
        },
        "costs": {
            "repeats": 1,
            "gpu": ["rnn-S-full", "transformer-S-full"],
            "cpu": ["lstm-S-full"],
            "tries": ["rnn-S"],
        },
    }
    generation = {
        "seed": 4213,
        "length": "median_train_review",
        "temperature_test": {
            "pile": "lm_valid",
            "reviews": 8,
            "prompt_tokens": 2,
            "grid": {"start": 0.8, "stop": 1.2, "step": 0.2},
            "limits": [0.6, 1.4],
            "models": ["kneser-ney", "rnn-S-full", "lstm-S-full", "transformer-S-full"],
        },
        "scores": {"pile": "lm_test", "reviews": 8, "prompt_tokens": 2, "bootstrap": 20},
        "models": ["kneser-ney", "rnn-S-full", "transformer-S-full", "lstm-own-frozen-s4213-full"],
        "samples_per_prompt": 3,
        "prompts": [{"name": "empty", "text": ""}, {"name": "two words", "text": "w0 w1"}],
        "report_body_prompts": [2],
    }
    eval_path, gen_path = root / "eval.yaml", root / "generation.yaml"
    eval_path.write_text(yaml.safe_dump(eval_config))
    gen_path.write_text(yaml.safe_dump(generation))
    return eval_path, gen_path


def test_evaluate_script_runs_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eval_path, gen_path = write_world(tmp_path)
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location("evaluate_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)  # dataclasses look their module up
    spec.loader.exec_module(module)
    assert module.main(["--config", str(eval_path), "--generation", str(gen_path)]) == 0

    results = tmp_path / "results" / "eval"
    test = json.loads((results / "test.json").read_text())
    assert test["checkpoints_reproduce_valid_J"] is True
    targets = {row["test"]["targets"] for row in test["models"].values()}
    assert targets == {test["targets"]}
    assert set(test["models"]) == {
        "kneser-ney",
        "add-k",
        "rnn-S-full",
        "lstm-S-full",
        "transformer-S-full",
        "lstm-own-frozen-s4213-full",
    }
    for row in test["models"].values():
        assert sum(g["targets"] for g in row["by_position"]) == test["targets"]

    costs = json.loads((results / "costs.json").read_text())
    assert costs["models"]["rnn-S-full"]["speed"]["gpu"]["matches_test_J"] is True
    assert "cpu" in costs["models"]["lstm-S-full"]["speed"]
    assert costs["tuning"]["rnn-S"]["count"] == 1
    assert costs["tuning"]["rnn-S"]["total_seconds"] == pytest.approx(4.0)  # try + final run
    assert costs["models"]["kneser-ney"]["speed"]["cpu"]["targets"] == test["targets"]

    generation = json.loads((results / "generation.json").read_text())
    assert 0.6 <= generation["temperature"] <= 1.4
    assert set(generation["scores"]) == {"real", *generation["prompts"]["1"]["models"]}
    assert generation["prompts"]["2"]["training_counts"]["w0"] > 0
    examples = (results / "examples_generation.txt").read_text()
    assert examples.count(">>") == 2 * 4  # one middle sample per prompt and model


def test_a_wrong_checkpoint_stops_the_script_before_the_test_pile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eval_path, gen_path = write_world(tmp_path)
    record_path = tmp_path / "results" / "runs" / "lstm-S-full" / "run.json"
    record = json.loads(record_path.read_text())
    record["summary"]["best_valid_J"] += 0.01  # as if the checkpoint came from another run
    record_path.write_text(json.dumps(record))
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location("evaluate_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit, match="lstm-S-full"):
        module.main(["--config", str(eval_path), "--generation", str(gen_path)])
    assert not (tmp_path / "results" / "eval").exists()
