"""Measure what the English-only rule removes, with samples to check by eye.

Usage:
    uv run python scripts/explore_language.py

The rule is in src/a1/data/language_filter.py. Writes to results/eda/:
    language_rule.json        every number
    examples_language.txt     random samples of what is removed and kept, and
                              Amazon markup examples (git-ignored, it quotes the data)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from a1.data.language_filter import KEEP, REASONS, english_only_reasons
from a1.data.review_table import (
    LENGTH_GROUPS,
    SHORT_MAX_WORDS,
    ReviewTable,
    build_review_table,
    fetch_excerpts,
    group_name,
    in_group,
)

DEFAULT_INPUT = Path("data/raw/All_Beauty.jsonl.gz")
DEFAULT_OUT = Path("results/eda")
SEED = 4213
REMOVED_SAMPLES = 40
KEPT_SAMPLES = 30


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="raw review file")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder")
    parser.add_argument("--seed", type=int, default=SEED, help="seed for the samples")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Missing {args.input}. Run: uv run python scripts/download_data.py", file=sys.stderr)
        return 1

    started = time.perf_counter()
    print("Reading, cleaning and identifying the language of every review...", flush=True)
    table = build_review_table(args.input)
    kept = table.kept_by_step_3()
    reasons = english_only_reasons(table)
    print(f"  done in {time.perf_counter() - started:.0f} s", flush=True)

    summary = _summarize(table, kept, reasons)
    rng = random.Random(args.seed)
    chosen = _choose_samples(table, kept, reasons, rng)
    excerpts = fetch_excerpts(args.input, {i for indices in chosen.values() for i in indices})
    samples = {
        name: [_describe(table, i, excerpts.get(i, "")) for i in indices]
        for name, indices in chosen.items()
    }
    samples |= {
        f"Amazon markup [[{name}: ...]]": examples
        for name, examples in sorted(table.markup_examples.items())
    }

    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "language_rule.json"
    examples_path = args.out / "examples_language.txt"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    examples_path.write_text(
        "\n\n".join(
            f"## {name}\n" + ("\n".join(f"- {line}" for line in lines) or "- (none)")
            for name, lines in samples.items()
        )
        + "\n",
        encoding="utf-8",
    )
    print("\n".join(_report(summary)))
    print()
    print(f"Done in {time.perf_counter() - started:.0f} s. Files in {args.out}/:")
    for path in (summary_path, examples_path):
        print(f"  {path.name}")
    return 0


def _summarize(table: ReviewTable, kept: np.ndarray, reasons: np.ndarray) -> dict:
    fasttext_alone = ~table.english()
    removed = kept & (reasons != KEEP)
    rows = []
    for low, high in [*LENGTH_GROUPS, (1, None)]:
        members = kept & in_group(table.words, low, high)
        rows.append(
            {
                "group": "all" if (low, high) == (1, None) else group_name(low, high),
                "reviews": int(members.sum()),
                "removed": int((members & removed).sum()),
                "by_rule": {
                    REASONS[code]: int((members & (reasons == code)).sum()) for code in REASONS
                },
                "fasttext_alone_would_remove": int((members & fasttext_alone).sum()),
                "fasttext_flags_now_kept": int((members & fasttext_alone & ~removed).sum()),
                "removed_but_not_flagged_by_fasttext": int(
                    (members & removed & ~fasttext_alone).sum()
                ),
            }
        )
    return {
        "rule": "src/a1/data/language_filter.py",
        "kept_by_step_3": int(kept.sum()),
        "removed": int(removed.sum()),
        "by_length_group": rows,
        "word_evidence_taught_by": {
            "english_reviews": table.evidence.english_reviews,
            "other_reviews": table.evidence.other_reviews,
        },
        "markup_reviews": dict(table.markup_reviews.most_common()),
    }


def _choose_samples(
    table: ReviewTable, kept: np.ndarray, reasons: np.ndarray, rng: random.Random
) -> dict[str, list[int]]:
    def pick(mask: np.ndarray, size: int) -> list[int]:
        indices = np.flatnonzero(mask).tolist()
        return sorted(rng.sample(indices, min(size, len(indices))))

    short = table.words <= SHORT_MAX_WORDS
    not_english_guess = ~table.english()
    chosen = {
        f"Removed: {REASONS[code]}": pick(kept & (reasons == code), REMOVED_SAMPLES)
        for code in REASONS
    }
    chosen["Kept: 1-6 words, though fastText's top guess is not English"] = pick(
        kept & (reasons == KEEP) & short & not_english_guess, KEPT_SAMPLES
    )
    chosen["Kept: 7+ words, though fastText's top guess is not English"] = pick(
        kept & (reasons == KEEP) & ~short & not_english_guess, KEPT_SAMPLES
    )
    return chosen


def _describe(table: ReviewTable, index: int, text: str) -> str:
    language = table.languages[table.language[index]]
    parts = [f"{table.rating[index]}*", f"fastText {language} {table.probability[index]:.2f}"]
    short = table.short_reviews.get(index)
    if short is not None:
        parts.append(f"word evidence {table.evidence.score(short.words):+.1f}")
    for sentence in table.foreign_sentences.get(index, [])[:1]:
        parts.append(
            f"foreign sentence ({sentence.language} {sentence.probability:.2f}, word evidence "
            f"{table.evidence.score(sentence.words):+.1f}): {sentence.text[:80]}"
        )
    return " | ".join([*parts, text])


def _report(summary: dict) -> list[str]:
    lines = [
        "== English-only rule ==",
        f"{summary['removed']:,} of {summary['kept_by_step_3']:,} reviews removed "
        f"({summary['removed'] / summary['kept_by_step_3']:.2%}).",
        "group | reviews | removed | by rule 1 / 2 / 3 | fastText alone would remove | "
        "of those, kept now | removed although fastText said English",
    ]
    for row in summary["by_length_group"]:
        by_rule = " / ".join(f"{count:,}" for count in row["by_rule"].values())
        lines.append(
            f"{row['group']:>11} | {row['reviews']:,} | {row['removed']:,} | {by_rule} | "
            f"{row['fasttext_alone_would_remove']:,} | {row['fasttext_flags_now_kept']:,} | "
            f"{row['removed_but_not_flagged_by_fasttext']:,}"
        )
    lines.append("rules: " + "; ".join(f"{code}) {name}" for code, name in REASONS.items()))
    evidence = summary["word_evidence_taught_by"]
    lines.append(
        f"Word evidence learnt from {evidence['english_reviews']:,} English and "
        f"{evidence['other_reviews']:,} non-English reviews of 7+ words."
    )
    markup = summary["markup_reviews"]
    lines += [
        "",
        "== Amazon markup [[NAME: ...]] ==",
        ", ".join(f"{name}: {count:,} reviews" for name, count in markup.items()) or "none found",
    ]
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
