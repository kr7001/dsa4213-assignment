"""Fit and score the n-gram models (D1.1).

Usage:
    uv run python scripts/train_ngram.py [--piles configs/piles.yaml] [--max-order 5] [--test]

It counts every n-gram of 1 to --max-order tokens in the language-model
training pile once, then for each order n scores two models on the validation
pile:

- Kneser-Ney: interpolated modified Kneser-Ney (Chen and Goodman, 1998), with
  nothing to tune;
- add-k: the course's interpolated add-k model, with k picked from a grid and
  the weights lambda fitted by EM, both on the validation pile (Lecture1_Demo,
  section 7).

It reports validation J and PPL for every model and order, the share of
validation n-grams never seen in training, and the cost: stored n-grams,
counting time and scoring speed. The best order of each model is picked on
validation. --test then scores those two models on the test pile, once: use it
only for the final numbers. Everything runs on the CPU and is written to
results/ngram/ngram.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from a1.data.sequences import encode_piles, load_piles
from a1.data.vocab import UNK_ID, Vocabulary
from a1.models.ngram import ADD_K_GRID, AddK, KneserNey, NgramCounts, unseen_shares

DEFAULT_CORPUS = Path("data/processed/reviews.jsonl.gz")
DEFAULT_PILES = Path("configs/piles.yaml")
VOCAB_PATH = Path("data/processed/vocab.json")
RESULTS = Path("results/ngram")


def score(model, reviews) -> dict[str, Any]:
    started = time.perf_counter()
    log_probs = model.log_probs(reviews)
    seconds = time.perf_counter() - started
    if log_probs.size != reviews.n_targets:
        raise AssertionError("The n-gram model scored a different number of targets")
    j = float(-log_probs.mean())
    return {
        "J": j,
        "PPL": math.exp(j),
        "targets": int(log_probs.size),
        "seconds": seconds,
        "targets_per_second": log_probs.size / seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--piles", type=Path, default=DEFAULT_PILES)
    parser.add_argument("--vocab", type=Path, default=VOCAB_PATH)
    parser.add_argument("--max-order", type=int, default=5)
    parser.add_argument("--test", action="store_true", help="score the chosen models on test")
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    if not args.vocab.exists():
        print(f"Missing {args.vocab}: run scripts/build_vocab.py first", file=sys.stderr)
        return 1

    vocab = Vocabulary.load(args.vocab)
    piles = load_piles(args.piles)
    wanted = {
        name: piles[name]
        for name in ("lm_train", "lm_valid", "lm_test")
        if name != "lm_test" or args.test
    }
    data = encode_piles(args.corpus, wanted, vocab)
    train, valid = data["lm_train"], data["lm_valid"]

    started = time.perf_counter()
    table = NgramCounts.build(train, len(vocab), args.max_order)
    counting_seconds = time.perf_counter() - started
    print(
        f"Counted n-grams of 1 to {args.max_order} tokens in {counting_seconds:.0f} s "
        f"({train.n_targets:,} training targets)"
    )

    rows: list[dict[str, Any]] = []
    for order in range(1, args.max_order + 1):
        kn = score(KneserNey(table, order), valid)
        rows.append(
            {
                "method": "kneser_ney",
                "order": order,
                "valid": kn,
                "stored_ngrams": table.stored_entries(order),
            }
        )
        best: dict[str, Any] | None = None
        for k in ADD_K_GRID:
            model = AddK(table, order, k)
            model.fit_weights(valid)
            result = score(model, valid)
            if best is None or result["J"] < best["valid"]["J"]:
                best = {
                    "method": "add_k",
                    "order": order,
                    "k": k,
                    "weights": model.weights.round(4).tolist(),
                    "valid": result,
                    "stored_ngrams": table.stored_entries(order),
                }
        rows.append(best)
        print(
            f"  n={order}: Kneser-Ney J {kn['J']:.4f} (PPL {kn['PPL']:.1f});  "
            f"add-k J {best['valid']['J']:.4f} (PPL {best['valid']['PPL']:.1f}, k={best['k']})"
        )

    chosen = {
        method: min((r for r in rows if r["method"] == method), key=lambda r: r["valid"]["J"])
        for method in ("kneser_ney", "add_k")
    }
    output: dict[str, Any] = {
        "vocabulary_size": len(vocab),
        "train_targets": train.n_targets,
        "valid_targets": valid.n_targets,
        "valid_unk_share": float((valid.ids == UNK_ID).sum()) / valid.n_targets,
        "counting_seconds": counting_seconds,
        "unseen_share_valid": unseen_shares(table, valid, args.max_order),
        "validation": rows,
        "chosen": {m: {"order": r["order"], "k": r.get("k")} for m, r in chosen.items()},
    }
    if args.test:
        test = data["lm_test"]
        output["test_unk_share"] = float((test.ids == UNK_ID).sum()) / test.n_targets
        output["test"] = {}
        for method, row in chosen.items():
            if method == "kneser_ney":
                model = KneserNey(table, row["order"])
            else:
                model = AddK(table, row["order"], row["k"], weights=np.asarray(row["weights"]))
            output["test"][method] = score(model, test)
            print(
                f"  test, {method} n={row['order']}: J {output['test'][method]['J']:.4f} "
                f"(PPL {output['test'][method]['PPL']:.1f})"
            )
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "ngram.json").write_text(json.dumps(output, indent=2) + "\n")
    shares = ", ".join(f"{n}: {s:.1%}" for n, s in output["unseen_share_valid"].items())
    print(f"Validation n-grams never seen in training: {shares}")
    print(f"Chosen on validation: {output['chosen']}. Wrote {args.results / 'ngram.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
