"""Describe the raw review file before any preprocessing decision is made.

Usage:
    uv run python scripts/explore_data.py

Writes to results/eda/:
    summary.json    every number, grouped by the decision it informs
    ratings.png, lengths.png, products.png
    examples.txt    review excerpts for reading by eye (git-ignored)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from a1.data.explore import collect_statistics, format_report, write_examples
from a1.data.explore_plots import plot_statistics

DEFAULT_INPUT = Path("data/raw/All_Beauty.jsonl.gz")
DEFAULT_OUT = Path("results/eda")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="raw review file")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder")
    parser.add_argument("--max-examples", type=int, default=10, help="excerpts kept per check")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Missing {args.input}. Run: uv run python scripts/download_data.py", file=sys.stderr)
        return 1

    started = time.perf_counter()
    try:
        stats = collect_statistics(args.input, max_examples=args.max_examples)
    except (EOFError, OSError) as error:
        print(
            f"Could not read {args.input} ({error}). The download may be incomplete; "
            "re-run: uv run python scripts/download_data.py --force",
            file=sys.stderr,
        )
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(stats.summary, indent=2) + "\n", encoding="utf-8")
    write_examples(stats, args.out / "examples.txt")
    charts = plot_statistics(stats, args.out)

    print(format_report(stats))
    print()
    print(f"Done in {time.perf_counter() - started:.0f} s. Files in {args.out}/:")
    for path in [summary_path, *charts, args.out / "examples.txt"]:
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
