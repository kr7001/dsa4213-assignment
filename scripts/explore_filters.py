"""Measure what the remaining data decisions need: English only, cap, duplicates.

Usage:
    uv run python scripts/explore_filters.py

Every review is cleaned as the models will see it, and the locked step 3 rules
are applied first (at least one word, at most 512 tokens). Writes to
results/eda/:
    filters.json            every number
    examples_filters.txt    samples to read by eye (git-ignored, it quotes the data)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from a1.data.duplicate_checks import format_duplicates, summarize_duplicates
from a1.data.language_checks import format_language, summarize_language
from a1.data.product_checks import format_products, summarize_products
from a1.data.review_table import MAX_TOKENS, build_review_table, fetch_excerpts

DEFAULT_INPUT = Path("data/raw/All_Beauty.jsonl.gz")
DEFAULT_OUT = Path("results/eda")
SEED = 4213


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="raw review file")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder")
    parser.add_argument("--seed", type=int, default=SEED, help="seed for samples and splits")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Missing {args.input}. Run: uv run python scripts/download_data.py", file=sys.stderr)
        return 1

    started = time.perf_counter()
    print("Reading, cleaning and identifying the language of every review...", flush=True)
    table = build_review_table(args.input)
    kept = table.kept_by_step_3()
    print(f"  done in {time.perf_counter() - started:.0f} s; measuring...", flush=True)

    language, language_samples = summarize_language(table, kept, args.seed)
    products = summarize_products(table, kept, args.seed)
    duplicates, duplicate_samples = summarize_duplicates(table, kept, args.seed)
    excerpts = fetch_excerpts(args.input, duplicate_samples.indices())

    step_3 = {
        "reviews": len(table),
        "removed_no_words": int((table.words == 0).sum()),
        f"removed_over_{MAX_TOKENS}_tokens": int(
            ((table.words > 0) & (table.tokens > MAX_TOKENS)).sum()
        ),
        "kept": int(kept.sum()),
    }
    summary = {
        "step_3": step_3,
        "language": language,
        "products": products,
        "duplicates": duplicates,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "filters.json"
    examples_path = args.out / "examples_filters.txt"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    samples = language_samples | duplicate_samples.render(excerpts)
    examples_path.write_text(
        "\n\n".join(
            f"## {name}\n" + ("\n".join(f"- {line}" for line in lines) or "- (none)")
            for name, lines in samples.items()
        )
        + "\n",
        encoding="utf-8",
    )

    report = [
        f"Step 3 rules: {step_3['reviews']:,} reviews, {step_3['removed_no_words']:,} with no "
        f"words and {step_3[f'removed_over_{MAX_TOKENS}_tokens']:,} over {MAX_TOKENS} tokens "
        f"removed, {step_3['kept']:,} kept.",
        "",
        *format_language(language),
        "",
        *format_products(products),
        "",
        *format_duplicates(duplicates),
    ]
    print("\n".join(report))
    print()
    print(f"Done in {time.perf_counter() - started:.0f} s. Files in {args.out}/:")
    for path in (summary_path, examples_path):
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
