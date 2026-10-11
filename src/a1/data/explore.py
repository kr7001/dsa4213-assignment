"""Corpus statistics behind every preprocessing decision.

One streaming pass over the raw review file collects the numbers the data
decisions rest on: corpus size, star ratings, reviews per product, review
lengths, repeated texts and raw-text noise. Nothing here changes the data; it
only describes it, so each later choice can point to a measured number.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from a1.data.text import (
    COMMON_ENGLISH_WORDS,
    NOISE_PATTERNS,
    course_tokenize,
    non_ascii_letter_share,
)

STAR_LEVELS = (1, 2, 3, 4, 5)
PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9)
SHORT_THRESHOLDS = (1, 3, 5, 10, 20, 50)  # share of reviews with fewer than k tokens
LONG_THRESHOLDS = (128, 256, 512, 1024)  # share of reviews with more than k tokens
CAP_VALUES = (1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1000)
AT_LEAST_VALUES = (2, 5, 10, 30, 100, 1000)
# Repeated texts are grouped by length: a short phrase written by many people
# ("love it") is a different case from a long review copied word for word.
DUPLICATE_LENGTH_BUCKETS: tuple[tuple[int, int | None], ...] = ((0, 4), (5, 19), (20, None))
NON_LATIN_LETTER_SHARE = 0.3
MIN_TOKENS_FOR_LANGUAGE_CHECK = 6
EXCERPT_CHARS = 160
TOP_REPEATED_TEXTS = 15


@dataclass
class _RepeatedText:
    """All reviews whose text is identical after lowercasing and joining whitespace."""

    count: int
    n_tokens: int
    excerpt: str
    products: set[int] = field(default_factory=set)
    ratings: Counter[int] = field(default_factory=Counter)


@dataclass
class CorpusStatistics:
    """Everything one pass over the corpus produces.

    Attributes:
        summary: Numbers only, safe to commit; saved as summary.json.
        lengths: Course-tokenizer tokens per review, shape (n_reviews,).
        ratings: Star rating per review, 1 to 5, shape (n_reviews,).
        reviews_per_product: Review count per product, shape (n_products,).
        examples: Review excerpts for each check. They quote the data, so they
            go to a file that git ignores.
    """

    summary: dict[str, Any]
    lengths: np.ndarray
    ratings: np.ndarray
    reviews_per_product: np.ndarray
    examples: dict[str, list[str]]


def iter_reviews(path: Path) -> Iterator[dict[str, Any]]:
    """Yield one review record per non-empty line of a JSON Lines file.

    Args:
        path: A ``.jsonl`` or gzip-compressed ``.jsonl.gz`` file.

    Yields:
        The parsed record of each review.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def collect_statistics(path: Path, max_examples: int = 10) -> CorpusStatistics:
    """Read the review file once and measure everything the data decisions need.

    Args:
        path: Raw review file in the Amazon Reviews 2023 format.
        max_examples: Excerpts kept per check, for reading by eye.

    Returns:
        The summary numbers, per-review arrays for plotting, and excerpts.
    """
    product_ids: dict[str, int] = {}
    product_of_review = array("i")
    lengths = array("i")
    ratings = array("b")
    user_counts: Counter[str] = Counter()
    year_counts: Counter[int] = Counter()
    noise_counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    first_seen: dict[bytes, tuple[int, int]] = {}  # text hash -> (product, rating)
    repeated: dict[bytes, _RepeatedText] = {}
    counts = Counter()

    def keep_example(name: str, excerpt: str) -> None:
        if len(examples[name]) < max_examples:
            examples[name].append(excerpt)

    for record in iter_reviews(path):
        text = record.get("text") or ""
        rating = int(round(float(record["rating"])))
        product_key = record.get("parent_asin") or record.get("asin") or "unknown"
        product = product_ids.setdefault(product_key, len(product_ids))
        tokens = course_tokenize(text)
        n_tokens = len(tokens)

        product_of_review.append(product)
        lengths.append(n_tokens)
        ratings.append(rating)
        user_counts[record.get("user_id", "")] += 1
        year = _year_of(record.get("timestamp"))
        if year is not None:
            year_counts[year] += 1
        counts["verified"] += bool(record.get("verified_purchase"))

        if not text.strip():
            counts["empty_text"] += 1
            counts["empty_text_with_title"] += bool((record.get("title") or "").strip())
        elif n_tokens == 0:
            counts["zero_tokens_nonempty_text"] += 1
            keep_example("zero_tokens", _excerpt(text))

        for name, pattern in NOISE_PATTERNS.items():
            match = pattern.search(text)
            if match:
                noise_counts[name] += 1
                keep_example(name, _excerpt(text, around=match.start()))

        if non_ascii_letter_share(text) >= NON_LATIN_LETTER_SHARE:
            counts["mostly_non_latin"] += 1
            keep_example("mostly_non_latin", _excerpt(text))
        elif n_tokens >= MIN_TOKENS_FOR_LANGUAGE_CHECK and COMMON_ENGLISH_WORDS.isdisjoint(tokens):
            counts["no_common_english_word"] += 1
            keep_example("no_common_english_word", _excerpt(text))

        normalized = " ".join(text.lower().split())
        if normalized:
            digest = hashlib.blake2b(normalized.encode("utf-8"), digest_size=12).digest()
            seen = first_seen.get(digest)
            if seen is None:
                first_seen[digest] = (product, rating)
            else:
                group = repeated.get(digest)
                if group is None:
                    group = _RepeatedText(count=1, n_tokens=n_tokens, excerpt=_excerpt(text))
                    group.products.add(seen[0])
                    group.ratings[seen[1]] += 1
                    repeated[digest] = group
                group.count += 1
                group.products.add(product)
                group.ratings[rating] += 1

    n_reviews = len(lengths)
    if n_reviews == 0:
        raise ValueError(f"No reviews found in {path}")

    lengths_np = np.asarray(lengths, dtype=np.int64)
    ratings_np = np.asarray(ratings, dtype=np.int64)
    reviews_per_product = np.bincount(
        np.asarray(product_of_review, dtype=np.int64), minlength=len(product_ids)
    )
    _add_repeated_examples(repeated, examples)

    summary = {
        "size": _size_summary(
            n_reviews, len(product_ids), user_counts, lengths_np, year_counts, counts["verified"]
        ),
        "ratings": _rating_summary(ratings_np),
        "products": _product_summary(reviews_per_product),
        "users": _user_summary(user_counts, n_reviews),
        "lengths": _length_summary(lengths_np, ratings_np, counts),
        "duplicates": _duplicate_summary(repeated, n_reviews),
        "noise": {
            name: {"reviews": noise_counts[name], "share": noise_counts[name] / n_reviews}
            for name in NOISE_PATTERNS
        },
        "language": {
            "mostly_non_latin_letters": counts["mostly_non_latin"],
            "no_common_english_word": counts["no_common_english_word"],
            "rule": (
                f"non-Latin: at least {NON_LATIN_LETTER_SHARE:.0%} of letters outside ASCII; "
                f"no common word: {MIN_TOKENS_FOR_LANGUAGE_CHECK}+ tokens and none of "
                f"{len(COMMON_ENGLISH_WORDS)} very common English words"
            ),
        },
    }
    return CorpusStatistics(
        summary=summary,
        lengths=lengths_np,
        ratings=ratings_np,
        reviews_per_product=reviews_per_product,
        examples=dict(examples),
    )


def _year_of(timestamp: Any) -> int | None:
    """Return the calendar year of a Unix timestamp given in ms (or seconds)."""
    if not timestamp:
        return None
    seconds = timestamp / 1000 if timestamp > 1e11 else timestamp
    return datetime.fromtimestamp(seconds, tz=UTC).year


def _excerpt(text: str, around: int = 0) -> str:
    """Return a single-line slice of ``text`` near position ``around``."""
    start = max(0, around - EXCERPT_CHARS // 3)
    piece = " ".join(text[start : start + EXCERPT_CHARS].split())
    prefix = "..." if start > 0 else ""
    suffix = "..." if start + EXCERPT_CHARS < len(text) else ""
    return f"{prefix}{piece}{suffix}"


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {f"p{p:g}": float(np.percentile(values, p)) for p in PERCENTILES}


def _size_summary(
    n_reviews: int,
    n_products: int,
    user_counts: Counter[str],
    lengths: np.ndarray,
    year_counts: Counter[int],
    n_verified: int,
) -> dict[str, Any]:
    return {
        "reviews": n_reviews,
        "products": n_products,
        "users": len(user_counts),
        "course_tokens": int(lengths.sum()),
        "years": {
            "first": min(year_counts) if year_counts else None,
            "last": max(year_counts) if year_counts else None,
            "per_year": {str(year): year_counts[year] for year in sorted(year_counts)},
        },
        "verified_purchase_share": n_verified / n_reviews,
    }


def _rating_summary(ratings: np.ndarray) -> dict[str, Any]:
    counts = {str(star): int((ratings == star).sum()) for star in STAR_LEVELS}
    return {
        "counts": counts,
        "shares": {star: count / len(ratings) for star, count in counts.items()},
    }


def _product_summary(reviews_per_product: np.ndarray) -> dict[str, Any]:
    total = int(reviews_per_product.sum())
    largest = int(reviews_per_product.max())
    ranked = np.sort(reviews_per_product)[::-1]

    def top_share(fraction: float) -> float:
        k = max(1, math.ceil(fraction * len(ranked)))
        return float(ranked[:k].sum() / total)

    cap_table = [
        {
            "cap": cap,
            "share_reviews_kept": float(np.minimum(reviews_per_product, cap).sum() / total),
            "products_above_cap": int((reviews_per_product > cap).sum()),
            "largest_product_after_cap": min(largest, cap),
        }
        for cap in CAP_VALUES
    ]
    return {
        "reviews_per_product_mean": total / len(reviews_per_product),
        "reviews_per_product_percentiles": _percentiles(reviews_per_product),
        "max_reviews_one_product": largest,
        "largest_product_share_of_reviews": largest / total,
        "share_products_with_one_review": float((reviews_per_product == 1).mean()),
        "products_with_at_least": {
            str(k): int((reviews_per_product >= k).sum()) for k in AT_LEAST_VALUES
        },
        "share_of_reviews_in_top_1pct_products": top_share(0.01),
        "share_of_reviews_in_top_10pct_products": top_share(0.10),
        "cap_table": cap_table,
    }


def _user_summary(user_counts: Counter[str], n_reviews: int) -> dict[str, Any]:
    per_user = np.fromiter(user_counts.values(), dtype=np.int64)
    return {
        "max_reviews_one_user": int(per_user.max()),
        "share_reviews_by_users_with_10_plus": float(per_user[per_user >= 10].sum() / n_reviews),
    }


def _length_summary(
    lengths: np.ndarray, ratings: np.ndarray, counts: Counter[str]
) -> dict[str, Any]:
    total_tokens = max(1, int(lengths.sum()))
    by_star = {}
    for star in STAR_LEVELS:
        subset = lengths[ratings == star]
        by_star[str(star)] = {
            "reviews": int(subset.size),
            "median": float(np.median(subset)) if subset.size else None,
            "mean": float(subset.mean()) if subset.size else None,
            "share_shorter_than": {
                str(k): float((subset < k).mean()) if subset.size else None
                for k in SHORT_THRESHOLDS
            },
            "share_longer_than": {
                str(k): float((subset > k).mean()) if subset.size else None for k in LONG_THRESHOLDS
            },
        }
    return {
        "unit": "course-tokenizer tokens on the raw text, before any cleaning",
        "mean": float(lengths.mean()),
        "percentiles": _percentiles(lengths),
        "max": int(lengths.max()),
        "empty_text": counts["empty_text"],
        "empty_text_with_title": counts["empty_text_with_title"],
        "zero_tokens_nonempty_text": counts["zero_tokens_nonempty_text"],
        "share_shorter_than": {str(k): float((lengths < k).mean()) for k in SHORT_THRESHOLDS},
        "share_longer_than": {str(k): float((lengths > k).mean()) for k in LONG_THRESHOLDS},
        "share_of_tokens_in_reviews_longer_than": {
            str(k): float(lengths[lengths > k].sum() / total_tokens) for k in LONG_THRESHOLDS
        },
        "by_star": by_star,
    }


def _bucket_label(low: int, high: int | None) -> str:
    return f"{low}-{high}" if high is not None else f"{low}+"


def _duplicate_summary(repeated: dict[bytes, _RepeatedText], n_reviews: int) -> dict[str, Any]:
    groups = list(repeated.values())
    by_length = []
    for low, high in DUPLICATE_LENGTH_BUCKETS:
        members = [g for g in groups if g.n_tokens >= low and (high is None or g.n_tokens <= high)]
        spanning = [g for g in members if len(g.products) > 1]
        by_length.append(
            {
                "tokens": _bucket_label(low, high),
                "distinct_texts": len(members),
                "reviews": sum(g.count for g in members),
                "extra_copies": sum(g.count - 1 for g in members),
                "texts_spanning_2plus_products": len(spanning),
                "reviews_in_texts_spanning_2plus_products": sum(g.count for g in spanning),
            }
        )
    extra_copies = sum(g.count - 1 for g in groups)
    return {
        "rule": "texts identical after lowercasing and joining whitespace",
        "distinct_repeated_texts": len(groups),
        "reviews_in_repeated_texts": sum(g.count for g in groups),
        "extra_copies": extra_copies,
        "extra_copies_share_of_reviews": extra_copies / n_reviews,
        "texts_spanning_2plus_products": sum(1 for g in groups if len(g.products) > 1),
        "texts_with_both_low_and_high_ratings": sum(
            1
            for g in groups
            if (g.ratings[1] + g.ratings[2]) > 0 and (g.ratings[4] + g.ratings[5]) > 0
        ),
        "by_length": by_length,
    }


def _add_repeated_examples(
    repeated: dict[bytes, _RepeatedText], examples: dict[str, list[str]]
) -> None:
    """Record the most repeated texts overall and among long texts."""

    def describe(group: _RepeatedText) -> str:
        return (
            f"{group.count:,} copies in {len(group.products):,} products, "
            f"{group.n_tokens} tokens: {group.excerpt}"
        )

    by_count = sorted(repeated.values(), key=lambda g: g.count, reverse=True)
    examples["repeated_most_common"] = [describe(g) for g in by_count[:TOP_REPEATED_TEXTS]]
    long_low = DUPLICATE_LENGTH_BUCKETS[-1][0]
    long_groups = [g for g in by_count if g.n_tokens >= long_low]
    examples["repeated_long"] = [describe(g) for g in long_groups[:TOP_REPEATED_TEXTS]]


def _num(value: float) -> str:
    """Format a statistic briefly: 12, 3.5, 658 (not 658.001)."""
    return f"{round(value, 1):g}"


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    if 0 < value < 0.001:
        return f"{value * 100:.3f}%"
    return f"{value * 100:.1f}%"


def format_report(stats: CorpusStatistics) -> str:
    """Render the summary as short plain text, grouped by the decision it informs."""
    s = stats.summary
    size, products, lengths, dup = s["size"], s["products"], s["lengths"], s["duplicates"]
    pp = products["reviews_per_product_percentiles"]
    lp = lengths["percentiles"]
    lines = [
        "== Size (pile sizes, subset) ==",
        f"reviews {size['reviews']:,} | products {size['products']:,} | users {size['users']:,}",
        f"course tokens {size['course_tokens']:,} | years {size['years']['first']} to "
        f"{size['years']['last']} | verified purchases {_pct(size['verified_purchase_share'])}",
        "",
        "== Star ratings (Part III labels) ==",
        "  ".join(
            f"{star}*: {count:,} ({_pct(s['ratings']['shares'][star])})"
            for star, count in s["ratings"]["counts"].items()
        ),
        "",
        "== Reviews per product (cap) ==",
        f"mean {products['reviews_per_product_mean']:.1f} | median {_num(pp['p50'])} | "
        f"90th pct {_num(pp['p90'])} | 99th pct {_num(pp['p99'])} | max "
        f"{products['max_reviews_one_product']:,} "
        f"({_pct(products['largest_product_share_of_reviews'])} of all reviews)",
        f"{_pct(products['share_products_with_one_review'])} of products have exactly 1 review; "
        f"top 1% of products hold {_pct(products['share_of_reviews_in_top_1pct_products'])} "
        f"of reviews, top 10% hold {_pct(products['share_of_reviews_in_top_10pct_products'])}",
        "products with at least k reviews: "
        + ", ".join(f"{k}: {n:,}" for k, n in products["products_with_at_least"].items()),
        "cap N -> reviews kept (products above the cap):",
    ]
    rows = products["cap_table"]
    for start in range(0, len(rows), 6):
        lines.append(
            "  "
            + " | ".join(
                f"{row['cap']}: {_pct(row['share_reviews_kept'])} ({row['products_above_cap']:,})"
                for row in rows[start : start + 6]
            )
        )
    lines += [
        "",
        "== Review length in course tokens (min and max length) ==",
        f"mean {lengths['mean']:.1f} | 5/25/50/75/95/99/99.9th pct: "
        + "/".join(_num(lp[key]) for key in ("p5", "p25", "p50", "p75", "p95", "p99", "p99.9"))
        + f" | max {lengths['max']:,}",
        f"empty text {lengths['empty_text']:,} (with a title: "
        f"{lengths['empty_text_with_title']:,}) | non-empty but 0 tokens "
        f"{lengths['zero_tokens_nonempty_text']:,}",
        "shorter than k: "
        + ", ".join(f"{k}: {_pct(v)}" for k, v in lengths["share_shorter_than"].items()),
        "longer than k: "
        + ", ".join(f"{k}: {_pct(v)}" for k, v in lengths["share_longer_than"].items())
        + " | share of all tokens in those reviews: "
        + ", ".join(
            f"{k}: {_pct(v)}" for k, v in lengths["share_of_tokens_in_reviews_longer_than"].items()
        ),
        "by star: median | shorter than 5 | shorter than 10 | longer than 256",
    ]
    for star, row in lengths["by_star"].items():
        shorter, longer = row["share_shorter_than"], row["share_longer_than"]
        median = "n/a" if row["median"] is None else _num(row["median"])
        lines.append(
            f"  {star}*: {median} | {_pct(shorter['5'])} | {_pct(shorter['10'])} "
            f"| {_pct(longer['256'])}"
        )
    lines += [
        "",
        "== Repeated texts (de-duplication) ==",
        f"{dup['reviews_in_repeated_texts']:,} reviews share their exact text with another; "
        f"keeping one copy each drops {dup['extra_copies']:,} "
        f"({_pct(dup['extra_copies_share_of_reviews'])})",
        f"{dup['texts_spanning_2plus_products']:,} of {dup['distinct_repeated_texts']:,} "
        f"repeated texts appear under 2+ products; "
        f"{dup['texts_with_both_low_and_high_ratings']:,} carry both 1-2* and 4-5* ratings",
    ]
    for bucket in dup["by_length"]:
        lines.append(
            f"  {bucket['tokens']} tokens: {bucket['distinct_texts']:,} texts, "
            f"{bucket['reviews']:,} reviews, {bucket['texts_spanning_2plus_products']:,} "
            "texts across 2+ products"
        )
    lines.append("most repeated:")
    lines += [f"  {line[:110]}" for line in stats.examples.get("repeated_most_common", [])[:5]]
    lines.append("most repeated with 20+ tokens:")
    lines += [f"  {line[:110]}" for line in stats.examples.get("repeated_long", [])[:3]]
    lines += ["", "== Raw-text noise (cleaning) =="]
    lines.append(
        " | ".join(
            f"{name}: {row['reviews']:,} ({_pct(row['share'])})" for name, row in s["noise"].items()
        )
    )
    language = s["language"]
    lines.append(
        f"possibly not English: {language['mostly_non_latin_letters']:,} mostly non-Latin, "
        f"{language['no_common_english_word']:,} with no common English word"
    )
    return "\n".join(lines)


def write_examples(stats: CorpusStatistics, path: Path) -> None:
    """Write the review excerpts behind each check to a git-ignored text file."""
    sections = []
    for name, excerpts in stats.examples.items():
        body = "\n".join(f"- {excerpt}" for excerpt in excerpts) or "- (none)"
        sections.append(f"## {name}\n{body}")
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
