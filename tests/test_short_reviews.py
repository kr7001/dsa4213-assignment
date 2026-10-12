"""End-to-end test of the short-review measurements on a tiny hand-made file.

Labelled reviews and their word counts after cleaning:
    positive: "Love it!" x2 (2), "good" x2 (1), "10/10" (0), thumbs-up emoji (0),
              "<br />Great &amp; cheap, 2.5 oz" (4)
    negative: "Love it!" (2), empty text (0), "Terrible. Broke after ..." (9)
    3 stars:  "It's ok" (2), left out of every positive/negative number
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from a1.data.short_reviews import (
    ZERO_WORD_KINDS,
    collect_short_review_statistics,
    format_short_report,
    sentiment_of,
    write_short_examples,
)

REVIEWS = [
    ("Love it!", 5.0),
    ("Love it!", 5.0),
    ("love it!", 1.0),  # same cleaned text, opposite label
    ("good", 4.0),
    ("Good", 5.0),
    ("10/10", 5.0),
    ("\U0001f44d", 5.0),
    ("", 1.0),
    ("Terrible. Broke after one week, would not buy again.", 1.0),
    ("It's ok", 3.0),
    ("<br />Great &amp; cheap, 2.5 oz", 4.0),
]


@pytest.fixture()
def review_file(tmp_path: Path) -> Path:
    path = tmp_path / "reviews.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for i, (text, rating) in enumerate(REVIEWS):
            record = {"rating": rating, "text": text, "parent_asin": f"P{i}", "user_id": "U"}
            handle.write(json.dumps(record) + "\n")
    return path


def _row(summary: dict, words: str) -> dict:
    return next(row for row in summary["by_word_count"] if row["words"] == words)


def test_sentiment_mapping() -> None:
    assert [sentiment_of(star) for star in (1, 2, 3, 4, 5)] == [0, 0, 2, 1, 1]


def test_counts_by_word_count(review_file: Path) -> None:
    summary = collect_short_review_statistics(review_file).summary
    assert summary["reviews"] == len(REVIEWS)
    assert _row(summary, "0")["reviews"] == 3
    assert _row(summary, "1")["reviews"] == 2
    assert _row(summary, "2")["reviews"] == 4
    assert _row(summary, "4")["reviews"] == 1
    assert _row(summary, "9")["reviews"] == 1
    assert _row(summary, "all")["reviews"] == len(REVIEWS)
    assert _row(summary, "2")["cumulative_share_reviews"] == pytest.approx(9 / 11)
    assert summary["zero_words"] == dict(zip(ZERO_WORD_KINDS, (1, 1, 1), strict=True))


def test_repeats_and_copy_agreement(review_file: Path) -> None:
    summary = collect_short_review_statistics(review_file).summary
    two_words = _row(summary, "2")
    # "love it !" x3 is repeated; "it's ok" is not.
    assert two_words["repeated_share"] == pytest.approx(3 / 4)
    # Each 5-star copy sees one positive and one negative other copy: a tie (0.5).
    # The 1-star copy sees two positive copies: wrong guess (0).
    assert two_words["copy_coverage"] == pytest.approx(1.0)
    assert two_words["copy_vote_accuracy"] == pytest.approx((0.5 + 0.5 + 0) / 3)
    assert two_words["copy_vote_right_on_negatives"] == pytest.approx(0.0)
    assert two_words["copy_vote_right_on_positives"] == pytest.approx(0.5)
    assert two_words["copy_vote_accuracy_50_50"] == pytest.approx(0.25)
    one_word = _row(summary, "1")
    assert one_word["copy_vote_accuracy"] == pytest.approx(1.0)  # "good" x2, both positive
    assert one_word["copy_vote_accuracy_50_50"] is None  # no negative to score


def test_minimum_word_rules(review_file: Path) -> None:
    summary = collect_short_review_statistics(review_file).summary
    rules = {rule["at_least_words"]: rule for rule in summary["minimum_words"]}
    assert rules[1]["share_reviews_removed"] == pytest.approx(3 / 11)
    assert rules[1]["share_negatives_removed"] == pytest.approx(1 / 3)
    assert rules[1]["share_positives_removed"] == pytest.approx(2 / 7)
    assert rules[3]["share_reviews_removed"] == pytest.approx(9 / 11)


def test_length_only_baseline(review_file: Path) -> None:
    rows = collect_short_review_statistics(review_file).summary["length_only_baseline"]
    at_least_one = rows[0]
    # Negatives [2, 9] against positives [2, 2, 1, 1, 4]: 9 beats all five, 2 beats
    # the two 1s and ties the two 2s, so (5 + 2 + 0.5 * 2) / (2 * 5) = 0.8.
    assert at_least_one["chance_negative_longer"] == pytest.approx(0.8)
    # "Positive if at most 4 words": all positives right, one negative of two right.
    assert at_least_one["best_accuracy"] == pytest.approx(0.75)
    assert at_least_one["rule"] == "positive if at most 4 words"


def test_report_and_examples(review_file: Path, tmp_path: Path) -> None:
    stats = collect_short_review_statistics(review_file)
    report = format_short_report(stats)
    assert "Reviews by number of words" in report and '"love it !" 3 (67%)' in report
    path = tmp_path / "examples_short_reviews.txt"
    write_short_examples(stats, path)
    text = path.read_text(encoding="utf-8")
    assert "tokens: <num> <num>" in text and "Terrible" not in text  # 9 words: not listed
