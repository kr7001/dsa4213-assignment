"""Measure what very short reviews look like, to decide the minimum length.

Usage:
    uv run python scripts/explore_short_reviews.py

Every review is cleaned and tokenized exactly as the models will see it
(src/a1/data/clean.py). Writes to results/eda/:
    short_reviews.json              every number
    examples_short_reviews.txt      common short texts and random examples
                                    (git-ignored, because it quotes the data)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from a1.data.short_reviews import (
    collect_short_review_statistics,
    format_short_report,
    write_short_examples,
)

DEFAULT_INPUT = Path("data/raw/All_Beauty.jsonl.gz")
DEFAULT_OUT = Path("results/eda")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="raw review file")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder")
    parser.add_argument("--examples", type=int, default=15, help="random reviews per group")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Missing {args.input}. Run: uv run python scripts/download_data.py", file=sys.stderr)
        return 1

    started = time.perf_counter()
    stats = collect_short_review_statistics(args.input, examples_per_group=args.examples)

    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "short_reviews.json"
    examples_path = args.out / "examples_short_reviews.txt"
    summary_path.write_text(json.dumps(stats.summary, indent=2) + "\n", encoding="utf-8")
    write_short_examples(stats, examples_path)

    print(format_short_report(stats))
    print()
    print(f"Done in {time.perf_counter() - started:.0f} s. Files in {args.out}/:")
    for path in (summary_path, examples_path):
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
