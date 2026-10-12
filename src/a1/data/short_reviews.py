"""Evidence for the minimum review length (step 3 of the data decisions).

One pass over the raw file cleans and tokenizes every review exactly as the
models will see it, then measures, for each number of words:

- how many reviews and tokens there are, and how positive the reviews are;
- how often the exact text is repeated (stock phrases versus unique writing);
- how often repeated copies agree on positive versus negative, which shows
  whether short reviews are easy points, ambiguous, or mislabelled.

It also measures how well review length alone separates positive from
negative reviews, the baseline Part III accuracy has to beat.

A word is a token that starts with a letter, so punctuation and "<num>" do not
count: "5 stars!!" has one word. Positive means 4 or 5 stars and negative 1 or
2 stars; 3-star reviews are left out of those numbers, as in Part III.
"""

from __future__ import annotations

import hashlib
import random
from array import array
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from a1.data.clean import clean_text, count_words, tokenize
from a1.data.explore import iter_reviews

WORD_BUCKETS: tuple[tuple[int, int | None], ...] = (
    *((k, k) for k in range(11)),
    (11, 20),
    (21, 50),
    (51, 100),
    (101, None),
)
LISTED_MAX_WORDS = 5  # reviews with up to this many words get top texts and examples
MINIMUM_WORDS = (1, 2, 3, 4, 5, 6, 11)  # "at least k words" rules to price
BASELINE_MINIMUMS = (1, 3, 5)  # Part III populations for the length-only baseline
MAX_TOKENS = 512
TOP_TEXTS = 10
SEED = 4213
EXCERPT_CHARS = 120

NEGATIVE, POSITIVE, NEUTRAL = 0, 1, 2
ZERO_WORD_KINDS = (
    "empty text",
    "nothing left after cleaning (emoji, symbols, other scripts)",
    "only punctuation or numbers",
)


@dataclass
class ShortReviewStatistics:
    """What one pass produces.

    Attributes:
        summary: Numbers only, safe to commit; saved as short_reviews.json.
        top_texts: For 0 to LISTED_MAX_WORDS words, the most common cleaned
            texts as (text, copies, share positive or None).
        examples: Random reviews per group, quoting the data, so they go to a
            file that git ignores.
    """

    summary: dict[str, Any]
    top_texts: dict[int, list[tuple[str, int, float | None]]]
    examples: dict[str, list[str]]


class _Reservoir:
    """Keep a uniform random sample of fixed size from a stream."""

    def __init__(self, size: int, rng: random.Random) -> None:
        self.size, self.rng, self.seen, self.items = size, rng, 0, []

    def offer(self, make_item: Any) -> None:
        """Consider one stream element; ``make_item`` builds it only if kept."""
        self.seen += 1
        if len(self.items) < self.size:
            self.items.append(make_item())
            return
        slot = self.rng.randrange(self.seen)
        if slot < self.size:
            self.items[slot] = make_item()


def sentiment_of(rating: int) -> int:
    """Map a star rating to NEGATIVE (1-2), NEUTRAL (3) or POSITIVE (4-5)."""
    if rating <= 2:
        return NEGATIVE
    return NEUTRAL if rating == 3 else POSITIVE


def collect_short_review_statistics(
    path: Path, examples_per_group: int = 15, seed: int = SEED
) -> ShortReviewStatistics:
    """Read the raw review file once and measure what short reviews look like.

    Args:
        path: Raw review file in the Amazon Reviews 2023 format.
        examples_per_group: Random reviews kept per word count and per kind of
            zero-word review.
        seed: Seed for the random examples.

    Returns:
        The summary numbers, the most common short texts and random examples.
    """
    rng = random.Random(seed)
    words, tokens, labels, text_of_review = array("i"), array("i"), array("b"), array("i")
    text_ids: dict[bytes, int] = {}
    words_of_text = array("i")
    short_texts: dict[int, str] = {}
    zero_word_kinds: Counter[str] = Counter()
    # Empty reviews have nothing to show, so they are counted but not sampled.
    groups = [*ZERO_WORD_KINDS[1:], *(_words_label(k) for k in range(1, LISTED_MAX_WORDS + 1))]
    reservoirs = {name: _Reservoir(examples_per_group, rng) for name in groups}

    for record in iter_reviews(path):
        raw = record.get("text") or ""
        rating = int(round(float(record["rating"])))
        review_tokens = tokenize(clean_text(raw))
        n_words = count_words(review_tokens)
        cleaned = " ".join(review_tokens)

        digest = hashlib.blake2b(cleaned.encode("utf-8"), digest_size=12).digest()
        text_id = text_ids.get(digest)
        if text_id is None:
            text_id = len(text_ids)
            text_ids[digest] = text_id
            words_of_text.append(n_words)
            if n_words <= LISTED_MAX_WORDS:
                short_texts[text_id] = cleaned

        words.append(n_words)
        tokens.append(len(review_tokens))
        labels.append(sentiment_of(rating))
        text_of_review.append(text_id)

        if n_words <= LISTED_MAX_WORDS:
            if n_words == 0:
                group = _zero_word_kind(raw, review_tokens)
                zero_word_kinds[group] += 1
            else:
                group = _words_label(n_words)
            if group in reservoirs:
                reservoirs[group].offer(
                    lambda raw=raw, rating=rating, cleaned=cleaned: (
                        f"{rating}* | {_excerpt(raw)} | tokens: {cleaned or '(none)'}"
                    )
                )

    if not words:
        raise ValueError(f"No reviews found in {path}")

    arrays = _Arrays(
        words=np.asarray(words, dtype=np.int64),
        tokens=np.asarray(tokens, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int64),
        text_ids=np.asarray(text_of_review, dtype=np.int64),
        n_texts=len(text_ids),
    )
    copies = _CopyStatistics.measure(arrays)
    summary = {
        "unit": (
            "words after the step 1 cleaning; a word starts with a letter, so punctuation and "
            "<num> are not words; tokens are all cleaned tokens"
        ),
        "reviews": int(arrays.words.size),
        "tokens": int(arrays.tokens.sum()),
        "distinct_cleaned_texts": arrays.n_texts,
        "by_word_count": _rows_by_word_count(arrays, copies),
        "minimum_words": _minimum_word_rules(arrays),
        "zero_words": {kind: zero_word_kinds[kind] for kind in ZERO_WORD_KINDS},
        "length_only_baseline": [
            _length_only_baseline(arrays, minimum) for minimum in BASELINE_MINIMUMS
        ],
        "long_reviews": _long_reviews(arrays),
    }
    top_texts = _top_texts(np.asarray(words_of_text, dtype=np.int64), short_texts, copies)
    examples = {name: reservoir.items for name, reservoir in reservoirs.items()}
    return ShortReviewStatistics(summary=summary, top_texts=top_texts, examples=examples)


@dataclass
class _Arrays:
    """One entry per review, plus the number of distinct cleaned texts."""

    words: np.ndarray
    tokens: np.ndarray
    labels: np.ndarray
    text_ids: np.ndarray
    n_texts: int


@dataclass
class _CopyStatistics:
    """How each review's exact cleaned text is spread over the corpus."""

    copies: np.ndarray  # per text: all reviews with that text
    negative_copies: np.ndarray  # per text: 1-2 star reviews with that text
    positive_copies: np.ndarray  # per text: 4-5 star reviews with that text
    repeated: np.ndarray  # per review: its text appears at least twice
    covered: np.ndarray  # per review: labelled, and another labelled copy exists
    credit: np.ndarray  # per review: 1 if the other copies' majority matches, 0.5 on a tie

    @classmethod
    def measure(cls, a: _Arrays) -> _CopyStatistics:
        copies = np.bincount(a.text_ids, minlength=a.n_texts)
        negative = np.bincount(a.text_ids[a.labels == NEGATIVE], minlength=a.n_texts)
        positive = np.bincount(a.text_ids[a.labels == POSITIVE], minlength=a.n_texts)
        is_negative, is_positive = a.labels == NEGATIVE, a.labels == POSITIVE
        # Leave the review itself out, then guess its label from the other copies.
        other_negative = negative[a.text_ids] - is_negative
        other_positive = positive[a.text_ids] - is_positive
        covered = (is_negative | is_positive) & (other_negative + other_positive > 0)
        guess_positive = other_positive > other_negative
        credit = np.where(
            other_positive == other_negative, 0.5, (guess_positive == is_positive).astype(float)
        )
        return cls(
            copies=copies,
            negative_copies=negative,
            positive_copies=positive,
            repeated=copies[a.text_ids] >= 2,
            covered=covered,
            credit=credit,
        )


def _words_label(n_words: int) -> str:
    return f"{n_words} word" if n_words == 1 else f"{n_words} words"


def _zero_word_kind(raw: str, review_tokens: list[str]) -> str:
    if not raw.strip():
        return ZERO_WORD_KINDS[0]
    return ZERO_WORD_KINDS[1] if not review_tokens else ZERO_WORD_KINDS[2]


def _excerpt(text: str) -> str:
    piece = " ".join(text.split())
    return piece if len(piece) <= EXCERPT_CHARS else piece[:EXCERPT_CHARS] + "..."


def _share(part: float, whole: float) -> float | None:
    return float(part / whole) if whole else None


def _bucket_name(low: int, high: int | None) -> str:
    if high is None:
        return f"{low}+"
    return str(low) if low == high else f"{low}-{high}"


def _row(a: _Arrays, c: _CopyStatistics, mask: np.ndarray, name: str) -> dict[str, Any]:
    labels = a.labels[mask]
    n_negative = int((labels == NEGATIVE).sum())
    n_positive = int((labels == POSITIVE).sum())
    labelled = mask & (a.labels != NEUTRAL)
    covered = mask & c.covered
    covered_negative = covered & (a.labels == NEGATIVE)
    covered_positive = covered & (a.labels == POSITIVE)
    right_on_negatives = _share(c.credit[covered_negative].sum(), covered_negative.sum())
    right_on_positives = _share(c.credit[covered_positive].sum(), covered_positive.sum())
    # Averaging the two rates gives the vote's accuracy on a 50/50 set, as in
    # Part III: 50% means the exact text says nothing beyond the majority label.
    vote_balanced = (
        None
        if right_on_negatives is None or right_on_positives is None
        else (right_on_negatives + right_on_positives) / 2
    )
    return {
        "words": name,
        "reviews": int(mask.sum()),
        "share_reviews": _share(mask.sum(), a.words.size),
        "share_tokens": _share(a.tokens[mask].sum(), a.tokens.sum()),
        "share_tokens_with_end_marker": _share(
            (a.tokens[mask] + 1).sum(), a.tokens.sum() + a.words.size
        ),
        "positive_share": _share(n_positive, n_negative + n_positive),
        "repeated_share": _share(c.repeated[mask].sum(), mask.sum()),
        "copy_coverage": _share(covered.sum(), labelled.sum()),
        "copy_vote_accuracy": _share(c.credit[covered].sum(), covered.sum()),
        "copy_vote_right_on_negatives": right_on_negatives,
        "copy_vote_right_on_positives": right_on_positives,
        "copy_vote_accuracy_50_50": vote_balanced,
    }


def _rows_by_word_count(a: _Arrays, c: _CopyStatistics) -> list[dict[str, Any]]:
    rows = []
    for low, high in WORD_BUCKETS:
        upper = a.words.max() if high is None else high
        mask = (a.words >= low) & (a.words <= upper)
        row = _row(a, c, mask, _bucket_name(low, high))
        row["cumulative_share_reviews"] = _share((a.words <= upper).sum(), a.words.size)
        rows.append(row)
    rows.append(_row(a, c, np.ones_like(a.words, dtype=bool), "all"))
    return rows


def _minimum_word_rules(a: _Arrays) -> list[dict[str, Any]]:
    """Price each "keep reviews with at least k words" rule."""
    negative, positive = a.labels == NEGATIVE, a.labels == POSITIVE
    rules = []
    for k in MINIMUM_WORDS:
        removed = a.words < k
        kept_negative = int((negative & ~removed).sum())
        kept_positive = int((positive & ~removed).sum())
        rules.append(
            {
                "at_least_words": k,
                "share_reviews_removed": _share(removed.sum(), a.words.size),
                "share_tokens_removed": _share(a.tokens[removed].sum(), a.tokens.sum()),
                "share_negatives_removed": _share((negative & removed).sum(), negative.sum()),
                "share_positives_removed": _share((positive & removed).sum(), positive.sum()),
                "positive_share_after": _share(kept_positive, kept_negative + kept_positive),
            }
        )
    return rules


def _length_only_baseline(a: _Arrays, minimum: int) -> dict[str, Any]:
    """How well length alone separates negative from positive reviews.

    Two numbers, both for a 50/50 set as in Part III:
    - the chance that a random negative review has more words than a random
      positive one (ties count half); 50% means length carries no signal;
    - the best accuracy of a rule "positive if at most t words" (or "if more
      than t words", whichever direction works), with t chosen to maximise it.
    """
    negative = a.words[(a.labels == NEGATIVE) & (a.words >= minimum)]
    positive = a.words[(a.labels == POSITIVE) & (a.words >= minimum)]
    result: dict[str, Any] = {
        "at_least_words": minimum,
        "negatives": int(negative.size),
        "positives": int(positive.size),
    }
    if negative.size == 0 or positive.size == 0:
        return result | {"chance_negative_longer": None, "best_accuracy": None, "rule": None}

    size = int(max(negative.max(), positive.max())) + 1
    negative_count = np.bincount(negative, minlength=size).astype(float)
    positive_count = np.bincount(positive, minlength=size).astype(float)
    positive_shorter = np.concatenate(([0.0], np.cumsum(positive_count)[:-1]))
    longer = (negative_count * positive_shorter).sum() + 0.5 * (
        negative_count * positive_count
    ).sum()
    chance_negative_longer = longer / (negative.size * positive.size)

    # Rule "positive if at most t words": right on positives with <= t words and
    # on negatives with > t words. Averaging the two rates gives 50/50 accuracy.
    positives_caught = np.cumsum(positive_count) / positive.size
    negatives_caught = 1 - np.cumsum(negative_count) / negative.size
    accuracy_short_positive = (positives_caught + negatives_caught) / 2
    t_short = int(np.argmax(accuracy_short_positive))
    t_long = int(np.argmin(accuracy_short_positive))
    if accuracy_short_positive[t_short] >= 1 - accuracy_short_positive[t_long]:
        best, rule = accuracy_short_positive[t_short], f"positive if at most {t_short} words"
    else:
        best, rule = 1 - accuracy_short_positive[t_long], f"positive if more than {t_long} words"
    return result | {
        "chance_negative_longer": float(chance_negative_longer),
        "best_accuracy": float(best),
        "rule": rule,
    }


def _long_reviews(a: _Arrays) -> dict[str, Any]:
    long = a.tokens > MAX_TOKENS
    return {
        "more_than_tokens": MAX_TOKENS,
        "reviews": int(long.sum()),
        "share_reviews": _share(long.sum(), a.words.size),
        "share_tokens": _share(a.tokens[long].sum(), a.tokens.sum()),
    }


def _top_texts(
    words_of_text: np.ndarray, short_texts: dict[int, str], c: _CopyStatistics
) -> dict[int, list[tuple[str, int, float | None]]]:
    top: dict[int, list[tuple[str, int, float | None]]] = {}
    for n_words in range(LISTED_MAX_WORDS + 1):
        candidates = np.flatnonzero(words_of_text == n_words)
        ranked = candidates[np.argsort(-c.copies[candidates], kind="stable")][:TOP_TEXTS]
        top[n_words] = [
            (
                short_texts[int(i)] or "(nothing left)",
                int(c.copies[i]),
                _share(c.positive_copies[i], c.negative_copies[i] + c.positive_copies[i]),
            )
            for i in ranked
        ]
    return top


def _pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    if 0 < value < 0.001:
        return f"{value * 100:.3f}%"
    return f"{value * 100:.{digits}f}%"


def format_short_report(stats: ShortReviewStatistics, top_per_row: int = 5) -> str:
    """Render the measurements as plain text, in the order the decision needs them."""
    s = stats.summary
    lines = [
        "== Reviews by number of words, after cleaning ==",
        f"{s['reviews']:,} reviews, {s['tokens']:,} tokens. Words leave out punctuation and <num>.",
        f"{'words':>6} {'reviews':>9} {'share':>7} {'up to here':>11} {'tokens':>7} "
        f"{'positive':>9} {'repeated':>9} {'text vote':>10} {'(covered)':>10}",
    ]
    for row in s["by_word_count"]:
        up_to_here = row.get("cumulative_share_reviews")
        lines.append(
            f"{row['words']:>6} {row['reviews']:>9,} {_pct(row['share_reviews']):>7} "
            f"{'' if up_to_here is None else _pct(up_to_here):>11} "
            f"{_pct(row['share_tokens'], 2):>7} {_pct(row['positive_share']):>9} "
            f"{_pct(row['repeated_share']):>9} {_pct(row['copy_vote_accuracy_50_50']):>10} "
            f"{'(' + _pct(row['copy_coverage']) + ')':>10}"
        )
    lines += [
        "up to here: share of reviews with at most this many words.",
        "tokens: share of all cleaned tokens.",
        "positive: share of 4-5* among 1-2* and 4-5* reviews (3* left out, as in Part III).",
        "repeated: share whose exact cleaned text appears at least twice in the corpus.",
        "text vote: guess each review's label by a vote of the other reviews with exactly",
        "  the same text, scored on a 50/50 set: 50% means the text tells you nothing,",
        "  100% means it gives the label away. (covered): share of the row's 1-2* and",
        "  4-5* reviews that have another copy to vote.",
        "",
        "== Keeping only reviews with at least k words would remove ==",
        f"{'k':>2} | {'reviews':>7} | {'tokens':>7} | {'negatives':>9} | {'positives':>9} | "
        "positive share after",
    ]
    for rule in s["minimum_words"]:
        lines.append(
            f"{rule['at_least_words']:>2} | {_pct(rule['share_reviews_removed']):>7} | "
            f"{_pct(rule['share_tokens_removed'], 2):>7} | "
            f"{_pct(rule['share_negatives_removed']):>9} | "
            f"{_pct(rule['share_positives_removed']):>9} | "
            f"{_pct(rule['positive_share_after'])}"
        )
    lines += [
        "",
        "== Reviews with no words ==",
        " | ".join(f"{kind}: {count:,}" for kind, count in s["zero_words"].items()),
        "",
        "== Most common short texts: copies (share positive) ==",
    ]
    for n_words, texts in stats.top_texts.items():
        shown = " | ".join(
            f'"{text}" {copies:,} ({_pct(positive, 0)})'
            for text, copies, positive in texts[:top_per_row]
        )
        lines.append(f"{n_words} {'word' if n_words == 1 else 'words'}: {shown or '(none)'}")
    lines += [
        "",
        "== Length alone as a sentiment guess, on a 50/50 set (Part III baseline) ==",
        f"{'k':>2} | {'negatives':>9} | {'positives':>9} | negative longer | best accuracy (rule)",
    ]
    for row in s["length_only_baseline"]:
        lines.append(
            f"{row['at_least_words']:>2} | {row['negatives']:>9,} | {row['positives']:>9,} | "
            f"{_pct(row['chance_negative_longer']):>15} | {_pct(row['best_accuracy'])} "
            f"({row['rule']})"
        )
    long = s["long_reviews"]
    lines += [
        "k: only reviews with at least k words. negative longer: chance a random negative",
        "  review has more words than a random positive one (ties count half); 50% would",
        "  mean length carries no signal.",
        "",
        "== Long reviews after cleaning ==",
        f"more than {long['more_than_tokens']} tokens: {long['reviews']:,} reviews "
        f"({_pct(long['share_reviews'], 2)}), {_pct(long['share_tokens'])} of tokens",
    ]
    return "\n".join(lines)


def write_short_examples(stats: ShortReviewStatistics, path: Path) -> None:
    """Write the most common short texts and random examples to a git-ignored file."""
    sections = []
    for n_words, texts in stats.top_texts.items():
        body = "\n".join(
            f"- {copies:,} {'copy' if copies == 1 else 'copies'} "
            f"({_pct(positive, 0)} positive): {text}"
            for text, copies, positive in texts
        )
        sections.append(f"## Most common texts with {_words_label(n_words)}\n{body or '- (none)'}")
    for name, items in stats.examples.items():
        body = "\n".join(f"- {item}" for item in items) or "- (none)"
        sections.append(f"## Random reviews: {name}\n{body}")
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
