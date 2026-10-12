"""Apply every locked data decision and write the clean corpus.

Usage:
    uv run python scripts/prepare_data.py

Steps, in order: cleaning (D0.4), lengths (D0.8), English only (D0.9),
duplicates (D0.11) and the cap per product (D0.10); see src/a1/data/prepare.py.

Writes:
    data/processed/reviews.jsonl.gz          the clean corpus (never committed)
    results/preprocess/report.json           counts after every step, by star, the
                                             cap check and the corpus fingerprint
    results/preprocess/examples_check.txt    random samples to check by hand
                                             (git-ignored, it quotes the data)

Then run scripts/check_corpus.py to check the corpus it wrote.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from a1.data.prepare import (
    SEED,
    choose_samples,
    format_report,
    prepare,
    render_samples,
    sample_indices,
    summarize,
    write_corpus,
)

DEFAULT_INPUT = Path("data/raw/All_Beauty.jsonl.gz")
DEFAULT_CORPUS = Path("data/processed/reviews.jsonl.gz")
DEFAULT_RESULTS = Path("results/preprocess")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="raw review file")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="clean corpus to write")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="report folder")
    parser.add_argument("--seed", type=int, default=SEED, help="seed for the cap and samples")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Missing {args.input}. Run: uv run python scripts/download_data.py", file=sys.stderr)
        return 1

    started = time.perf_counter()
    print("Step 1 of 2: reading, cleaning and checking every review (a few minutes)...", flush=True)
    prepared = prepare(args.input, seed=args.seed)
    samples = choose_samples(prepared)
    print("Step 2 of 2: writing the clean corpus...", flush=True)
    written = write_corpus(args.input, prepared, args.corpus, sample_indices(samples))

    summary = summarize(prepared, written)
    args.results.mkdir(parents=True, exist_ok=True)
    report_path = args.results / "report.json"
    examples_path = args.results / "examples_check.txt"
    report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    rendered = render_samples(prepared, samples, written.excerpts)
    examples_path.write_text(
        "\n\n".join(
            f"## {title}\n" + ("\n".join(f"- {line}" for line in lines) or "- (none)")
            for title, lines in rendered.items()
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("\n".join(format_report(summary)))
    print()
    print(f"Done in {time.perf_counter() - started:.0f} s. Wrote:")
    for path in (args.corpus, report_path, examples_path):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
