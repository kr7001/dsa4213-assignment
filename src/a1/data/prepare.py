"""The data pipeline: every locked data decision, in order, on the raw file.

1. Cleaning (D0.4): see clean.py.
2. Lengths (D0.8): keep reviews with at least one word and at most 512 tokens.
3. English only (D0.9): see language_filter.py.
4. Duplicates (D0.11): see dedup.py.
5. Cap per product (D0.10): see cap.py. The cap is computed from the ratings
   of the reviews left after step 4. If ratings of the same product are no
   more alike than ratings of different products (correlation 0 or below),
   extra reviews never stop adding information, so no cap is applied. The cap
   is also recomputed on random halves of the products, to show it does not
   depend on which products end up in the test pile.

The corpus is written with a fixed gzip header, and the report records the
SHA-256 of its uncompressed content, so a re-run can be checked as identical.

Every product also gets a fixed random split key in [0, 1), from a hash of
its id and the seed. The piles (language-model training, validation and test,
and the Part III sets) are cut from these keys later, once their sizes are
decided, without running this pipeline again.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from a1.data import dedup, language_filter
from a1.data.cap import (
    INFORMATION_SHARE,
    cap_mask,
    cap_on_random_halves,
    design_effect_cap,
    information,
)
from a1.data.clean import clean_text, tokenize
from a1.data.explore import iter_reviews
from a1.data.product_checks import intraclass_correlation
from a1.data.review_table import (
    LENGTH_GROUPS,
    MAX_TOKENS,
    SHORT_MAX_WORDS,
    ReviewTable,
    build_review_table,
    excerpt,
    group_name,
    in_group,
)

SEED = 4213
STAR_LEVELS = (1, 2, 3, 4, 5)
SAMPLES = {"language removed": 40, "language kept": 30, "duplicates": 10}


@dataclass
class Prepared:
    """The table and what each step decided.

    Attributes:
        table: Per-review facts from one pass over the raw file.
        stages: Boolean mask of the reviews kept after each step, in order.
        language: Per review, KEEP (0) or the English-only rule that removes it.
        duplicate: Per review, KEEP (0) or the duplicate rule that removes it.
        duplicate_of: For a review that repeats an earlier text (removed, or
            kept as a short copy by another person), that earlier review; else -1.
        rho: Intraclass correlation of star ratings within products (after
            step 4), or None if it cannot be computed.
        cap: Most reviews kept per product, or None for no cap.
        seed: Seed for the cap, the split keys and the samples.
        cap_check: rho and the cap recomputed on random halves of the products,
            or None when there is no cap.
    """

    table: ReviewTable
    stages: dict[str, np.ndarray]
    language: np.ndarray
    duplicate: np.ndarray
    duplicate_of: np.ndarray
    rho: float | None
    cap: int | None
    seed: int
    cap_check: dict[str, Any] | None = None

    @property
    def final(self) -> np.ndarray:
        return self.stages["after cap"]


def prepare(path: Path, seed: int = SEED) -> Prepared:
    """Run steps 1 to 5 on the raw review file.

    Args:
        path: Raw review file in the Amazon Reviews 2023 format.
        seed: Seed for the cap's random choice.

    Returns:
        What every step kept and why the rest went.
    """
    table = build_review_table(path)
    stages = {"raw": np.ones(len(table), dtype=bool)}
    stages["after lengths"] = table.kept_by_step_3()
    language = language_filter.english_only_reasons(table)
    stages["after English only"] = stages["after lengths"] & (language == language_filter.KEEP)
    duplicate, duplicate_of = dedup.duplicate_reasons(table, stages["after English only"], seed)
    stages["after duplicates"] = stages["after English only"] & (duplicate == dedup.KEEP)
    kept = stages["after duplicates"]
    rho = intraclass_correlation(table.rating[kept], table.product[kept])
    cap_check = None
    if rho is not None and 0 < rho < 1:
        cap: int | None = design_effect_cap(rho)
        stages["after cap"] = cap_mask(table.product, kept, cap, seed)
        cap_check = cap_on_random_halves(table.rating[kept], table.product[kept], seed)
    else:
        cap = None
        stages["after cap"] = kept.copy()
    return Prepared(table, stages, language, duplicate, duplicate_of, rho, cap, seed, cap_check)


def split_key(product: str, seed: int = SEED) -> float:
    """A fixed pseudo-random number in [0, 1) for a product id."""
    digest = hashlib.blake2b(f"{seed}:{product}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


@dataclass
class CorpusWritten:
    """What :func:`write_corpus` wrote.

    Attributes:
        rows: Reviews written.
        sha256: SHA-256 of the uncompressed content, the same value that
            ``gunzip -c reviews.jsonl.gz | shasum -a 256`` prints.
        excerpts: Excerpts of the reviews the hand-check samples quote, by index.
    """

    rows: int
    sha256: str
    excerpts: dict[int, str]


def write_corpus(
    raw_path: Path, prepared: Prepared, out_path: Path, quote: set[int]
) -> CorpusWritten:
    """Write the kept reviews, cleaned and tokenized, one JSON object per line.

    Each line holds the raw file position ("id"), product, user, time, stars,
    the product's split key and the cleaned tokens joined by single spaces.
    The gzip header carries no time or file name, so the same data always
    gives the same file.

    Args:
        raw_path: The raw review file the table was built from.
        prepared: Output of :func:`prepare`.
        out_path: A ``.jsonl.gz`` file to write.
        quote: Review indices whose excerpts the hand-check samples need.

    Returns:
        The row count, the fingerprint and the excerpts.
    """
    table, final = prepared.table, prepared.final
    keys: dict[int, float] = {}
    excerpts: dict[int, str] = {}
    digest = hashlib.sha256()
    rows = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        out_path.open("wb") as file,
        gzip.GzipFile(filename="", mode="wb", fileobj=file, mtime=0) as handle,
    ):
        for index, record in enumerate(iter_reviews(raw_path)):
            raw = record.get("text") or ""
            if index in quote:
                excerpts[index] = excerpt(raw)
            if not final[index]:
                continue
            product = int(table.product[index])
            if product not in keys:
                keys[product] = split_key(table.product_names[product], prepared.seed)
            row = {
                "id": index,
                "product": table.product_names[product],
                "user": table.user_names[int(table.user[index])],
                "time": int(table.timestamp[index]),
                "stars": int(table.rating[index]),
                "split_key": keys[product],
                "text": " ".join(tokenize(clean_text(raw))),
            }
            line = (json.dumps(row) + "\n").encode("utf-8")
            handle.write(line)
            digest.update(line)
            rows += 1
    return CorpusWritten(rows=rows, sha256=digest.hexdigest(), excerpts=excerpts)


def _counts(table: ReviewTable, mask: np.ndarray) -> dict[str, Any]:
    stars = {str(s): int((mask & (table.rating == s)).sum()) for s in STAR_LEVELS}
    positive = stars["4"] + stars["5"]
    labelled = positive + stars["1"] + stars["2"]
    return {
        "reviews": int(mask.sum()),
        "tokens": int(table.tokens[mask].sum()),
        "products": int(np.unique(table.product[mask]).size),
        "by_stars": stars,
        "positive_share": positive / labelled if labelled else None,
    }


def _by_group(
    table: ReviewTable, before: np.ndarray, reasons: np.ndarray, names: dict[int, str]
) -> list[dict[str, Any]]:
    rows = []
    for low, high in LENGTH_GROUPS:
        members = before & in_group(table.words, low, high)
        rows.append(
            {
                "group": group_name(low, high),
                "reviews": int(members.sum()),
                "removed": int((members & (reasons != 0)).sum()),
                "by_rule": {
                    name: int((members & (reasons == c)).sum()) for c, name in names.items()
                },
            }
        )
    return rows


def summarize(prepared: Prepared, corpus: CorpusWritten | None = None) -> dict[str, Any]:
    """Every number the report needs, by step, plus the corpus fingerprint if written."""
    table, stages = prepared.table, prepared.stages
    names = list(stages)
    lengths_in, language_in = stages["raw"], stages["after lengths"]
    duplicates_in, cap_in = stages["after English only"], stages["after duplicates"]
    removed_language = language_in & (prepared.language != 0)
    language_names = Counter(table.languages[i] for i in table.language[removed_language])
    product_counts = np.bincount(table.product[cap_in])
    final = prepared.final
    return {
        "seed": prepared.seed,
        "stages": [{"stage": name} | _counts(table, stages[name]) for name in names],
        "lengths": {
            "removed_no_words": int((lengths_in & (table.words == 0)).sum()),
            f"removed_over_{MAX_TOKENS}_tokens": int(
                (lengths_in & (table.words > 0) & (table.tokens > MAX_TOKENS)).sum()
            ),
        },
        "english_only": {
            "removed": int(removed_language.sum()),
            "share_removed": float(removed_language.sum() / language_in.sum()),
            "by_rule": {
                name: int((language_in & (prepared.language == code)).sum())
                for code, name in language_filter.REASONS.items()
            },
            "by_length_group": _by_group(
                table, language_in, prepared.language, language_filter.REASONS
            ),
            "fasttext_top_guess_of_removed": dict(language_names.most_common(8)),
            "word_score_learnt_from": {
                "english_reviews": table.evidence.english_reviews,
                "other_reviews": table.evidence.other_reviews,
            },
        },
        "duplicates": {
            "removed": int((duplicates_in & (prepared.duplicate != 0)).sum()),
            "by_rule": {
                name: int((duplicates_in & (prepared.duplicate == code)).sum())
                for code, name in dedup.REASONS.items()
            },
            "by_length_group": _by_group(table, duplicates_in, prepared.duplicate, dedup.REASONS),
        },
        "cap": _cap_summary(prepared, product_counts, cap_in, final),
        "final": _counts(table, final) | {"users": int(np.unique(table.user[final]).size)},
        "markup_reviews": dict(table.markup_reviews.most_common()),
        "email_addresses_removed_from_reviews": table.email_reviews,
        "corpus": None
        if corpus is None
        else {"rows": corpus.rows, "sha256_uncompressed": corpus.sha256},
    }


def _cap_summary(
    prepared: Prepared, product_counts: np.ndarray, cap_in: np.ndarray, final: np.ndarray
) -> dict[str, Any]:
    rho, cap = prepared.rho, prepared.cap
    has_cap = cap is not None and rho is not None
    return {
        "icc_stars": rho,
        "information_limit": 1 / rho if has_cap else None,
        "information_share": INFORMATION_SHARE,
        "cap": cap,
        "information_at_cap": information(cap, rho) if has_cap else None,
        "products_above_cap": int((product_counts > cap).sum()) if has_cap else 0,
        "removed": int((cap_in & ~final).sum()),
        "share_kept": float(final.sum() / cap_in.sum()),
        "check_random_halves": prepared.cap_check,
    }


@dataclass
class _Sample:
    title: str
    rows: list[tuple[int, int]]  # (review, its original or -1)
    verb: str = "removed"  # how the first review of a pair is labelled


def choose_samples(prepared: Prepared) -> list[_Sample]:
    """Pick the random reviews to check by hand (indices only)."""
    table, stages = prepared.table, prepared.stages
    rng = random.Random(prepared.seed)

    def pick(mask: np.ndarray, size: int) -> list[int]:
        indices = np.flatnonzero(mask).tolist()
        return sorted(rng.sample(indices, min(size, len(indices))))

    language_in = stages["after lengths"]
    short = table.words <= SHORT_MAX_WORDS
    not_english_guess = ~table.english()
    samples = [
        _Sample(
            f"English only, removed: {name}",
            [
                (i, -1)
                for i in pick(
                    language_in & (prepared.language == code), SAMPLES["language removed"]
                )
            ],
        )
        for code, name in language_filter.REASONS.items()
    ]
    kept = language_in & (prepared.language == 0) & not_english_guess
    samples += [
        _Sample(
            "English only, kept although fastText guessed another language: 1-6 words",
            [(i, -1) for i in pick(kept & short, SAMPLES["language kept"])],
        ),
        _Sample(
            "English only, kept although fastText guessed another language: 7+ words",
            [(i, -1) for i in pick(kept & ~short, SAMPLES["language kept"])],
        ),
    ]
    duplicates_in = stages["after English only"]
    for code in (dedup.SAME_PERSON_NEAR_COPY, dedup.OTHER_PERSON_COPY):
        chosen = pick(duplicates_in & (prepared.duplicate == code), SAMPLES["duplicates"])
        samples.append(
            _Sample(
                f"Duplicates, removed: {dedup.REASONS[code]}",
                [(i, int(prepared.duplicate_of[i])) for i in chosen],
            )
        )
    # The other side of the copy rule: exact copies by another person below the
    # 14-word line are kept, because people write common sentences independently.
    low, high = dedup.LONG_MIN_WORDS, dedup.OTHER_PERSON_MIN_WORDS - 1
    kept_copies = (
        duplicates_in
        & (prepared.duplicate == dedup.KEEP)
        & (prepared.duplicate_of >= 0)
        & in_group(table.words, low, high)
    )
    chosen = pick(kept_copies, SAMPLES["duplicates"])
    samples.append(
        _Sample(
            f"Duplicates, kept: exact copy of {low}-{high} words by another person",
            [(i, int(prepared.duplicate_of[i])) for i in chosen],
            verb="kept",
        )
    )
    return samples


def sample_indices(samples: list[_Sample]) -> set[int]:
    return {i for sample in samples for pair in sample.rows for i in pair if i >= 0}


def render_samples(
    prepared: Prepared, samples: list[_Sample], excerpts: dict[int, str]
) -> dict[str, list[str]]:
    """Turn the chosen samples into readable lines."""
    table = prepared.table
    evidence = table.evidence

    def language_line(i: int) -> str:
        parts = [
            f"{table.rating[i]}*",
            f"fastText {table.languages[table.language[i]]} {table.probability[i]:.2f}",
        ]
        short = table.short_reviews.get(i)
        words = short.words if short is not None else table.flagged_long_words.get(i)
        if words is not None:
            parts.append(f"word score {evidence.score(words):+.1f}")
        for sentence in table.foreign_sentences.get(i, [])[:1]:
            parts.append(
                f"foreign sentence ({sentence.language} {sentence.probability:.2f}, average "
                f"word score {evidence.mean_score(sentence.words):+.1f}): {sentence.text[:80]}"
            )
        return " | ".join([*parts, excerpts.get(i, "")])

    rendered: dict[str, list[str]] = {}
    for sample in samples:
        lines = []
        for review, source in sample.rows:
            if source < 0:
                lines.append(language_line(review))
            else:
                lines.append(
                    f"{table.rating[review]}* | {sample.verb}: {excerpts.get(review, '')}\n"
                    f"    copy of ({table.rating[source]}*): {excerpts.get(source, '')}"
                )
        rendered[sample.title] = lines
    for name, examples in sorted(table.markup_examples.items()):
        rendered[f"Amazon markup [[{name}: ...]] as written"] = examples
    return rendered


def format_report(summary: dict[str, Any]) -> list[str]:
    """Render the summary as plain-text report lines."""
    lines = ["== Reviews after each step ==", "step | reviews | tokens | products | positive share"]
    for row in summary["stages"]:
        positive = row["positive_share"]
        lines.append(
            f"{row['stage']:>18} | {row['reviews']:,} | {row['tokens']:,} | "
            f"{row['products']:,} | {'n/a' if positive is None else f'{positive:.1%}'}"
        )
    lengths = summary["lengths"]
    lines += [
        "",
        f"Lengths: {lengths['removed_no_words']:,} with no words, "
        f"{lengths[f'removed_over_{MAX_TOKENS}_tokens']:,} over {MAX_TOKENS} tokens.",
    ]
    english = summary["english_only"]
    lines.append(
        f"English only: {english['removed']:,} removed ({english['share_removed']:.2%}); "
        + "; ".join(f"{name}: {count:,}" for name, count in english["by_rule"].items())
    )
    for row in english["by_length_group"]:
        lines.append(f"  {row['group']:>11}: {row['removed']:,} of {row['reviews']:,}")
    lines.append(
        "  fastText's top guess for the removed: "
        + ", ".join(f"{lang} {n:,}" for lang, n in english["fasttext_top_guess_of_removed"].items())
    )
    duplicates = summary["duplicates"]
    lines.append(
        f"Duplicates: {duplicates['removed']:,} removed; "
        + "; ".join(f"{name}: {count:,}" for name, count in duplicates["by_rule"].items())
    )
    cap = summary["cap"]
    if cap["cap"] is None:
        lines.append(
            "Cap: none, because ratings of one product are no more alike than ratings of "
            "different products (correlation "
            + ("n/a" if cap["icc_stars"] is None else f"{cap['icc_stars']:.3f}")
            + ")."
        )
    else:
        lines += [
            f"Cap: star ratings within a product correlate at {cap['icc_stars']:.3f}, so a "
            f"product's information levels off at {cap['information_limit']:.2f} reviews; "
            f"{cap['information_share']:.0%} of that needs {cap['cap']} reviews.",
            f"  {cap['products_above_cap']:,} products above the cap; {cap['removed']:,} "
            f"reviews removed; {cap['share_kept']:.1%} of reviews kept.",
        ]
        check = cap["check_random_halves"]
        if check is not None and check["cap"]["min"] is not None:
            lines.append(
                f"  Check: on {check['repeats']} random halves of the products the cap ranges "
                f"from {check['cap']['min']:.0f} to {check['cap']['max']:.0f} "
                f"(median {check['cap']['median']:.0f})."
            )
    lines.append("")
    final = summary["final"]
    stars = ", ".join(f"{s}*: {n:,}" for s, n in final["by_stars"].items())
    lines += [
        f"Final corpus: {final['reviews']:,} reviews, {final['products']:,} products, "
        f"{final['users']:,} users, {final['tokens']:,} tokens.",
        f"  {stars}",
        "Amazon markup found: "
        + ", ".join(f"{name} in {n:,} reviews" for name, n in summary["markup_reviews"].items()),
        f"Email addresses removed from {summary['email_addresses_removed_from_reviews']:,} "
        "reviews.",
    ]
    corpus = summary.get("corpus")
    if corpus is not None:
        fingerprint = corpus["sha256_uncompressed"]
        lines.append(f"Corpus fingerprint (SHA-256 of the uncompressed file): {fingerprint}")
    return lines
