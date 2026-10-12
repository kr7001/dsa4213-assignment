"""Evidence for the English-only filter.

Measures what fastText calls each review, by length group and confidence, and
how the word evidence judges short reviews. The random samples it writes are
for reading by eye, which is how we estimate how often each rule is right.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from a1.data.review_table import (
    LENGTH_GROUPS,
    SHORT_MAX_WORDS,
    ReviewTable,
    group_name,
    in_group,
)

BANDS: tuple[tuple[float, float], ...] = ((0.0, 0.5), (0.5, 0.9), (0.9, 1.01))
EVIDENCE_THRESHOLDS = (1.0, 2.0, 3.0)
MAIN_EVIDENCE_THRESHOLD = 2.0  # used for the agreement table and samples
SAMPLES = 25
TOP_TEXTS = 30
TOP_LANGUAGES = 12
TOP_WORDS = 25


def band_name(low: float, high: float) -> str:
    return f"{low:.1f} to {min(high, 1.0):.1f}"


@dataclass
class _ShortText:
    """All short reviews whose prepared text is identical."""

    copies: int
    language: str
    probability: float
    evidence: float
    excerpt: str


def summarize_language(
    table: ReviewTable, kept: np.ndarray, seed: int
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Measure language evidence on the reviews kept by step 3.

    Args:
        table: The review table.
        kept: Boolean mask of reviews kept so far.
        seed: Seed for the random samples.

    Returns:
        Numbers for the summary file, and text samples for the examples file.
    """
    rng = random.Random(seed)
    english = table.english()
    summary: dict[str, Any] = {
        "identifier": "fastText lid.176, compressed model bundled with fast-langdetect",
        "reviews": int(kept.sum()),
        "not_english": int((kept & ~english).sum()),
        "top_languages": _top_languages(table, kept),
        "by_length_group": [],
    }
    for low, high in LENGTH_GROUPS:
        members = kept & in_group(table.words, low, high)
        flagged = members & ~english
        summary["by_length_group"].append(
            {
                "group": group_name(low, high),
                "reviews": int(members.sum()),
                "not_english": int(flagged.sum()),
                "not_english_by_probability": {
                    band_name(lo, hi): int(
                        (flagged & (table.probability >= lo) & (table.probability < hi)).sum()
                    )
                    for lo, hi in BANDS
                },
            }
        )

    short = _short_texts(table, kept)
    summary["word_evidence"] = _evidence_summary(table, kept, short)
    samples = _samples(table, kept, english, short, rng)
    return summary, samples


def _top_languages(table: ReviewTable, kept: np.ndarray) -> list[dict[str, Any]]:
    counts = np.bincount(table.language[kept], minlength=len(table.languages))
    order = np.argsort(-counts, kind="stable")
    total = int(kept.sum())
    rows = [
        {"language": table.languages[i], "reviews": int(counts[i]), "share": counts[i] / total}
        for i in order[:TOP_LANGUAGES]
        if counts[i]
    ]
    rest = int(counts[order[TOP_LANGUAGES:]].sum())
    if rest:
        rows.append({"language": "other", "reviews": rest, "share": rest / total})
    return rows


def _short_texts(table: ReviewTable, kept: np.ndarray) -> dict[str, _ShortText]:
    """Group the kept short reviews by their prepared text."""
    groups: dict[str, _ShortText] = {}
    for index, review in table.short_reviews.items():
        if not kept[index]:
            continue
        group = groups.get(review.text)
        if group is None:
            groups[review.text] = _ShortText(
                copies=1,
                language=table.languages[table.language[index]],
                probability=float(table.probability[index]),
                evidence=table.evidence.score(review.words),
                excerpt=review.excerpt,
            )
        else:
            group.copies += 1
    return groups


def _evidence_summary(
    table: ReviewTable, kept: np.ndarray, short: dict[str, _ShortText]
) -> dict[str, Any]:
    evidence = table.evidence
    by_group = []
    for low, high in (g for g in LENGTH_GROUPS if g[1] is not None and g[1] <= SHORT_MAX_WORDS):
        n_reviews, fasttext_flags = 0, 0
        flagged = Counter()
        both = Counter()
        for index, review in table.short_reviews.items():
            if not kept[index] or not low <= table.words[index] <= high:
                continue
            n_reviews += 1
            score = short[review.text].evidence
            not_english = table.languages[table.language[index]] != "en"
            fasttext_flags += not_english
            for threshold in EVIDENCE_THRESHOLDS:
                if score >= threshold:
                    flagged[threshold] += 1
            is_foreign = score >= MAIN_EVIDENCE_THRESHOLD
            both[(not_english, is_foreign)] += 1
        by_group.append(
            {
                "group": group_name(low, high),
                "reviews": n_reviews,
                "fasttext_not_english": fasttext_flags,
                "words_foreign_at_threshold": {f"{t:g}": flagged[t] for t in EVIDENCE_THRESHOLDS},
                f"agreement_at_{MAIN_EVIDENCE_THRESHOLD:g}": {
                    "both": both[(True, True)],
                    "fasttext_only": both[(True, False)],
                    "words_only": both[(False, True)],
                    "neither": both[(False, False)],
                },
            }
        )
    return {
        "taught_by": "reviews of 7+ words that fastText labels with probability 0.9 or more",
        "english_reviews": evidence.english_reviews,
        "other_reviews": evidence.other_reviews,
        "min_reviews_per_word": evidence.min_reviews,
        "short_reviews": by_group,
    }


def _samples(
    table: ReviewTable,
    kept: np.ndarray,
    english: np.ndarray,
    short: dict[str, _ShortText],
    rng: random.Random,
) -> dict[str, list[str]]:
    samples: dict[str, list[str]] = {}

    def describe(index: int) -> str:
        return (
            f"{table.rating[index]}* | {table.languages[table.language[index]]} "
            f"{table.probability[index]:.2f}, then "
            f"{table.languages[table.runner_up[index]]} {table.runner_up_probability[index]:.2f}"
            f" | {table.flagged_excerpts.get(index, '')}"
        )

    flagged = [i for i in table.flagged_excerpts if kept[i] and not english[i]]
    for low, high in LENGTH_GROUPS:
        for lo, hi in BANDS:
            cell = [
                i
                for i in flagged
                if table.words[i] >= low
                and (high is None or table.words[i] <= high)
                and lo <= table.probability[i] < hi
            ]
            chosen = rng.sample(cell, min(SAMPLES, len(cell)))
            name = (
                f"fastText not English: {group_name(low, high)}, "
                f"top probability {band_name(lo, hi)}"
            )
            samples[name] = [describe(i) for i in sorted(chosen)]

    def describe_text(text: str, group: _ShortText) -> str:
        return (
            f"{group.copies:,} copies | fastText {group.language} {group.probability:.2f} | "
            f"word evidence {group.evidence:+.1f} | {group.excerpt}"
        )

    by_copies = sorted(short.items(), key=lambda item: -item[1].copies)
    threshold = MAIN_EVIDENCE_THRESHOLD
    lists = {
        "short texts fastText calls not English": [
            item for item in by_copies if item[1].language != "en"
        ],
        f"short texts with word evidence {threshold:g}+": [
            item for item in by_copies if item[1].evidence >= threshold
        ],
        "short texts fastText calls not English but words do not": [
            item for item in by_copies if item[1].language != "en" and item[1].evidence < threshold
        ],
        "short texts the words call foreign but fastText does not": [
            item for item in by_copies if item[1].language == "en" and item[1].evidence >= threshold
        ],
    }
    for name, items in lists.items():
        samples[f"Most common {name}"] = [describe_text(t, g) for t, g in items[:TOP_TEXTS]]

    # Random short reviews (not distinct texts), so frequent texts count by their copies.
    short_indices = [i for i in table.short_reviews if kept[i]]
    bands = {
        f"word evidence {threshold:g}+": lambda s: s >= threshold,
        f"word evidence 1 to {threshold:g}": lambda s: 1 <= s < threshold,
    }
    for name, test in bands.items():
        cell = [i for i in short_indices if test(short[table.short_reviews[i].text].evidence)]
        chosen = rng.sample(cell, min(SAMPLES, len(cell)))
        samples[f"Random short reviews with {name}"] = [
            f"{table.rating[i]}* | fastText {table.languages[table.language[i]]} "
            f"{table.probability[i]:.2f} | word evidence "
            f"{short[table.short_reviews[i].text].evidence:+.1f} | {table.short_reviews[i].excerpt}"
            for i in sorted(chosen)
        ]

    evidence = table.evidence
    words = [w for w in set(evidence.english_counts) | set(evidence.other_counts)]
    scored = sorted(((evidence.word_score(w), w) for w in words), reverse=True)
    scored = [(s, w) for s, w in scored if s != 0.0]

    def describe_word(score: float, word: str) -> str:
        return (
            f"{word}: {score:+.1f} (in {evidence.other_counts[word]:,} non-English and "
            f"{evidence.english_counts[word]:,} English reviews)"
        )

    samples["Most foreign-typical words"] = [describe_word(s, w) for s, w in scored[:TOP_WORDS]]
    samples["Most English-typical words"] = [
        describe_word(s, w) for s, w in scored[-TOP_WORDS:][::-1]
    ]
    return samples


def format_language(summary: dict[str, Any]) -> list[str]:
    """Render the language numbers as report lines."""
    lines = [
        "== Language (English-only filter) ==",
        f"{summary['reviews']:,} reviews kept by step 3; fastText calls "
        f"{summary['not_english']:,} of them not English "
        f"({summary['not_english'] / summary['reviews']:.2%}).",
        "Top guesses: "
        + ", ".join(
            f"{row['language'] or 'none'} {row['reviews']:,} ({row['share']:.2%})"
            for row in summary["top_languages"]
        ),
        "Not English by length group and fastText's confidence in its top guess:",
    ]
    for row in summary["by_length_group"]:
        bands = row["not_english_by_probability"]
        lines.append(
            f"  {row['group']:>11}: {row['not_english']:,} of {row['reviews']:,} "
            f"({row['not_english'] / max(1, row['reviews']):.2%}) | "
            + " | ".join(f"{name}: {count:,}" for name, count in bands.items())
        )
    evidence = summary["word_evidence"]
    lines += [
        f"Word evidence learnt from {evidence['english_reviews']:,} English and "
        f"{evidence['other_reviews']:,} non-English reviews ({evidence['taught_by']}).",
        "Short reviews flagged (fastText not English | word evidence 1+ / 2+ / 3+ | "
        f"agreement at {MAIN_EVIDENCE_THRESHOLD:g}: both, fastText only, words only):",
    ]
    for row in evidence["short_reviews"]:
        words = row["words_foreign_at_threshold"]
        agree = row[f"agreement_at_{MAIN_EVIDENCE_THRESHOLD:g}"]
        lines.append(
            f"  {row['group']:>11}: {row['fasttext_not_english']:,} | "
            f"{words['1']:,} / {words['2']:,} / {words['3']:,} | "
            f"{agree['both']:,}, {agree['fasttext_only']:,}, {agree['words_only']:,} "
            f"(of {row['reviews']:,})"
        )
    return lines
