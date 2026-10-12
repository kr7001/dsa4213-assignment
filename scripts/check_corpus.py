"""Check the clean corpus written by scripts/prepare_data.py.

Usage:
    uv run python scripts/check_corpus.py

Runs every check in src/a1/data/corpus_checks.py on the corpus the models will
read: structure and agreement with report.json (including the fingerprint),
duplicates (including the evidence for the 14-word copy rule) and leftovers
(the stated limitations, measured).

Writes:
    results/preprocess/corpus_checks.json            every number
    results/preprocess/examples_corpus_checks.txt    near-identical pairs to read
                                                     (git-ignored, it quotes the data)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from a1.data.corpus_checks import example_lines, format_checks, load_corpus, run_checks

DEFAULT_CORPUS = Path("data/processed/reviews.jsonl.gz")
DEFAULT_RESULTS = Path("results/preprocess")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="clean corpus")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="report folder")
    args = parser.parse_args()

    if not args.corpus.exists():
        print(f"Missing {args.corpus}. Run: uv run python scripts/prepare_data.py", file=sys.stderr)
        return 1
    report_path = args.results / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else None

    started = time.perf_counter()
    print("Reading the corpus...", flush=True)
    corpus = load_corpus(args.corpus)
    print("Running the checks (a few minutes)...", flush=True)
    results, pairs = run_checks(corpus, report)

    args.results.mkdir(parents=True, exist_ok=True)
    checks_path = args.results / "corpus_checks.json"
    examples_path = args.results / "examples_corpus_checks.txt"
    checks_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    examples_path.write_text(
        "\n\n".join(
            f"## {title}\n" + ("\n".join(f"- {line}" for line in lines) or "- (none)")
            for title, lines in example_lines(corpus, pairs).items()
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("\n".join(format_checks(results)))
    print()
    print(f"Done in {time.perf_counter() - started:.0f} s. Wrote:")
    for path in (checks_path, examples_path):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
