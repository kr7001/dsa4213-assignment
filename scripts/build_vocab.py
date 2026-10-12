"""Build the shared vocabulary from the language-model training pile and describe every pile.

Usage:
    uv run python scripts/build_vocab.py [--piles configs/piles.yaml]

The vocabulary keeps every token seen at least 5 times in the training pile
(D0.4) and is shared by every model (P1.1). It is saved to
data/processed/vocab.json, which stays out of git because it comes from the
reviews. results/data/piles.json gets the size of each pile (reviews,
targets) and its share of <unk> targets, which is reported next to every
perplexity.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from a1.data.sequences import count_tokens, encode_piles, load_piles
from a1.data.vocab import UNK_ID, Vocabulary

DEFAULT_CORPUS = Path("data/processed/reviews.jsonl.gz")
DEFAULT_PILES = Path("configs/piles.yaml")
VOCAB_PATH = Path("data/processed/vocab.json")
RESULTS = Path("results/data")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--piles", type=Path, default=DEFAULT_PILES)
    parser.add_argument("--vocab", type=Path, default=VOCAB_PATH)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    if not args.corpus.exists():
        print(f"Missing {args.corpus}: run scripts/prepare_data.py first", file=sys.stderr)
        return 1

    started = time.perf_counter()
    piles = load_piles(args.piles)
    counts, train_reviews = count_tokens(args.corpus, piles["lm_train"])
    vocab = Vocabulary.build(counts)
    vocab.save(args.vocab)
    encoded = encode_piles(args.corpus, piles, vocab)
    summary = {
        "vocabulary_size": len(vocab),
        "min_count": vocab.min_count,
        "piles": {
            name: {
                "split_keys": [piles[name].low, piles[name].high],
                "reviews": len(reviews),
                "targets": reviews.n_targets,
                "unk_share": float((reviews.ids == UNK_ID).sum()) / max(1, reviews.n_targets),
            }
            for name, reviews in encoded.items()
        },
    }
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "piles.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(
        f"|V| = {len(vocab):,} (tokens seen {vocab.min_count}+ times in lm_train, plus 4 special)"
    )
    for name, row in summary["piles"].items():
        low, high = row["split_keys"]
        print(
            f"  {name:<12} keys [{low:.2f}, {high:.2f})  {row['reviews']:>8,} reviews  "
            f"{row['targets']:>11,} targets  <unk> {row['unk_share']:.2%}"
        )
    print(
        f"Done in {time.perf_counter() - started:.0f} s. Wrote {args.vocab} and "
        f"{args.results / 'piles.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
