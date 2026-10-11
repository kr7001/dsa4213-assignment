"""End-to-end test of the corpus statistics on a tiny hand-made file."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from a1.data.explore import collect_statistics, format_report, write_examples
from a1.data.explore_plots import plot_statistics

LONG_COPY = (
    "This serum absorbs quickly and leaves my skin soft, and after two weeks the dry "
    "patches on my cheeks are gone, so I will buy it again."
)


def _record(text: str, rating: float, product: str, user: str) -> dict:
    return {
        "rating": rating,
        "title": "t",
        "text": text,
        "images": [],
        "asin": product,
        "parent_asin": product,
        "user_id": user,
        "timestamp": 1588687728923,
        "helpful_vote": 0,
        "verified_purchase": True,
    }


@pytest.fixture()
def review_file(tmp_path: Path) -> Path:
    records = [
        _record("Love it!", 5.0, "P1", "U1"),
        _record("love   it!", 5.0, "P2", "U2"),  # same text after normalising
        _record(LONG_COPY, 5.0, "P1", "U3"),
        _record(LONG_COPY, 4.0, "P3", "U3"),  # long copy under another product
        _record("Broke after a week.<br />Would not buy again.", 1.0, "P1", "U4"),
        _record("Tom &amp; Jerry says it&#39;s ok", 3.0, "P1", "U5"),
        _record("", 2.0, "P4", "U6"),
    ]
    path = tmp_path / "reviews.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def test_counts(review_file: Path) -> None:
    summary = collect_statistics(review_file).summary
    assert summary["size"]["reviews"] == 7
    assert summary["size"]["products"] == 4
    assert summary["ratings"]["counts"] == {"1": 1, "2": 1, "3": 1, "4": 1, "5": 3}
    assert summary["products"]["max_reviews_one_product"] == 4
    assert summary["lengths"]["empty_text"] == 1
    assert summary["noise"]["html_tag"]["reviews"] == 1
    assert summary["noise"]["html_entity"]["reviews"] == 1


def test_repeated_texts(review_file: Path) -> None:
    duplicates = collect_statistics(review_file).summary["duplicates"]
    assert duplicates["distinct_repeated_texts"] == 2
    assert duplicates["extra_copies"] == 2
    assert duplicates["texts_spanning_2plus_products"] == 2
    long_bucket = duplicates["by_length"][-1]
    assert long_bucket["distinct_texts"] == 1 and long_bucket["reviews"] == 2


def test_cap_table_keeps_less_as_the_cap_shrinks(review_file: Path) -> None:
    table = collect_statistics(review_file).summary["products"]["cap_table"]
    kept = [row["share_reviews_kept"] for row in table]
    assert kept == sorted(kept)
    assert table[0]["share_reviews_kept"] == pytest.approx(4 / 7)  # cap 1: one per product


def test_report_examples_and_charts(review_file: Path, tmp_path: Path) -> None:
    stats = collect_statistics(review_file)
    report = format_report(stats)
    assert "Star ratings" in report and "Repeated texts" in report
    write_examples(stats, tmp_path / "examples.txt")
    assert "Broke after a week" in (tmp_path / "examples.txt").read_text(encoding="utf-8")
    for chart in plot_statistics(stats, tmp_path / "charts"):
        assert chart.stat().st_size > 0
