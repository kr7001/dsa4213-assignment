"""Measure what each slice of products holds, to choose the pile sizes (D0.4, D3.1).

Usage:
    uv run python scripts/explore_piles.py

Every product has a random split key in [0, 1), and a pile is a range of keys.
This script cuts the key range into 20 slices of 0.05 and reports, for each:

- reviews and targets (tokens + <eos>), the units of language-model cost;
- star ratings, and the size of a balanced Part III set (D3.1: 1-2 stars
  negative, 4-5 positive, 3 left out; each class sampled down to the smaller);
- labelled reviews by length group (D0.8: 1-2, 3-6, 7-20, 21+ words);
- the standard error of J on that slice, under a simple unigram model built
  from keys [0, 0.5) with the shared vocabulary rule (5+ counts). This shows
  how precisely a validation or test pile of that size measures J. Reviews
  are the unit of resampling, because tokens of one review are not
  independent: SE(J) = sqrt(sum_i (L_i - J n_i)^2) / sum_i n_i, the delta
  method for a ratio, where L_i is review i's total negative log-likelihood
  and n_i its number of targets.

It writes results/data/pile_sizes.json. No pile is fixed here; the numbers feed
the decision.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from a1.data.clean import count_words
from a1.data.vocab import MIN_COUNT

DEFAULT_CORPUS = Path("data/processed/reviews.jsonl.gz")
RESULTS = Path("results/data")
SLICES = 20
FIRST_EVAL_KEY = 0.5  # keys below this are language-model training (test run rule)
LENGTH_GROUPS = ((1, 2), (3, 6), (7, 20), (21, 10**9))


def length_group(n_words: int) -> str:
    for low, high in LENGTH_GROUPS:
        if low <= n_words <= high:
            return f"{low}-{high}" if high < 10**9 else f"{low}+"
    return "0"


def slice_of(key: float) -> int:
    return min(int(key * SLICES), SLICES - 1)


def read_corpus(path: Path) -> list[tuple[int, int, list[str]]]:
    """(slice, stars, tokens) for every review."""
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows.append((slice_of(row["split_key"]), int(row["stars"]), row["text"].split(" ")))
    return rows


def unigram_model(rows: list[tuple[int, int, list[str]]]) -> dict[str, float]:
    """Negative log-probabilities of a unigram model fitted on keys below 0.5.

    Tokens seen fewer than MIN_COUNT times share the <unk> entry; every review
    adds one <eos>. Every target then has a nonzero probability (add-one on
    the kept entries keeps it proper even if <unk> were empty).
    """
    counts: Counter = Counter()
    reviews = 0
    for part, _, tokens in rows:
        if part < FIRST_EVAL_KEY * SLICES:
            counts.update(tokens)
            reviews += 1
    kept = {t: n for t, n in counts.items() if n >= MIN_COUNT}
    unk = sum(n for t, n in counts.items() if n < MIN_COUNT)
    table = {**kept, "<unk>": unk, "<eos>": reviews}
    total = sum(table.values()) + len(table)
    return {t: -math.log((n + 1) / total) for t, n in table.items()}


def describe_slices(rows: list[tuple[int, int, list[str]]]) -> list[dict]:
    nll = unigram_model(rows)
    unk_nll, eos_nll = nll["<unk>"], nll["<eos>"]
    by_slice: list[list[tuple[int, list[str]]]] = [[] for _ in range(SLICES)]
    for part, rating, tokens in rows:
        by_slice[part].append((rating, tokens))
    slices = []
    for part in range(SLICES):
        stars: Counter = Counter()
        groups: dict[str, Counter] = {"negative": Counter(), "positive": Counter()}
        review_nll, review_targets = [], []
        for rating, tokens in by_slice[part]:
            stars[rating] += 1
            label = "negative" if rating <= 2 else "positive" if rating >= 4 else None
            if label:
                groups[label][length_group(count_words(tokens))] += 1
            review_nll.append(sum(nll.get(t, unk_nll) for t in tokens) + eos_nll)
            review_targets.append(len(tokens) + 1)
        losses = np.asarray(review_nll)
        targets = np.asarray(review_targets, dtype=np.float64)
        j = float(losses.sum() / targets.sum())
        se = float(np.sqrt(((losses - j * targets) ** 2).sum()) / targets.sum())
        negatives = stars[1] + stars[2]
        positives = stars[4] + stars[5]
        slices.append(
            {
                "keys": [round(part / SLICES, 4), round((part + 1) / SLICES, 4)],
                "reviews": int(targets.size),
                "targets": int(targets.sum()),
                "stars": {str(k): stars[k] for k in range(1, 6)},
                "negative": negatives,
                "positive": positives,
                "balanced_part3": 2 * min(negatives, positives),
                "labelled_by_length": {k: dict(v) for k, v in groups.items()},
                "unigram_J": j,
                "unigram_J_standard_error": se,
            }
        )
    return slices


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    if not args.corpus.exists():
        print(f"Missing {args.corpus}: run scripts/prepare_data.py first", file=sys.stderr)
        return 1
    started = time.perf_counter()
    rows = read_corpus(args.corpus)
    slices = describe_slices(rows)
    summary = {
        "slices": slices,
        "note": "unigram J is a yardstick for precision only, not a Part I result",
        "seconds": time.perf_counter() - started,
    }
    args.results.mkdir(parents=True, exist_ok=True)
    out = args.results / "pile_sizes.json"
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")

    print(
        f"{'keys':<12}{'reviews':>9}{'targets':>11}{'neg':>8}{'pos':>8}{'balanced':>10}{'SE(J)':>9}"
    )
    for s in slices:
        low, high = s["keys"]
        print(
            f"[{low:.2f},{high:.2f})".ljust(12)
            + f"{s['reviews']:>9,}{s['targets']:>11,}{s['negative']:>8,}{s['positive']:>8,}"
            + f"{s['balanced_part3']:>10,}{s['unigram_J_standard_error']:>9.4f}"
        )
    print(f"Wrote {out} in {summary['seconds']:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
