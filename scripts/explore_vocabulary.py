"""Measure the vocabulary choices on the clean corpus.

Usage:
    uv run python scripts/explore_vocabulary.py [--glove FILE ...]

For each vocabulary size it measures how much text from unseen products would
become <unk>, building on one half of the products and testing on the other.
It counts contractions, and with --glove it measures how much of the vocabulary
public GloVe vectors cover, with contractions kept whole or split as GloVe's
tokenizer splits them. A GloVe file can be a Stanford .txt or .zip
(for example glove.2024.wikigiga.50d.zip) or a gensim-data .gz.

Writes results/eda/vocabulary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from a1.data.vocabulary_checks import format_vocabulary, summarize_vocabulary

DEFAULT_CORPUS = Path("data/processed/reviews.jsonl.gz")
DEFAULT_RESULTS = Path("results/eda")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="clean corpus")
    parser.add_argument("--glove", type=Path, nargs="*", default=[], help="GloVe files")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="output folder")
    args = parser.parse_args()

    missing = [path for path in [args.corpus, *args.glove] if not path.exists()]
    if missing:
        print("Missing: " + ", ".join(map(str, missing)), file=sys.stderr)
        return 1

    started = time.perf_counter()
    print("Counting tokens in both halves of the products (a minute or two)...", flush=True)
    summary = summarize_vocabulary(args.corpus, args.glove)
    args.results.mkdir(parents=True, exist_ok=True)
    out = args.results / "vocabulary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print()
    print("\n".join(format_vocabulary(summary)))
    print()
    print(f"Done in {time.perf_counter() - started:.0f} s. Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
