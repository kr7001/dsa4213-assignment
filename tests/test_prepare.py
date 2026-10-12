"""Tests for the cap, the duplicate rules and the whole data pipeline."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from a1.data import dedup
from a1.data.cap import cap_mask, design_effect_cap, information
from a1.data.prepare import (
    choose_samples,
    format_report,
    prepare,
    render_samples,
    sample_indices,
    split_key,
    summarize,
    write_corpus,
)
from a1.data.review_table import build_review_table

# 24 words for building long test reviews.
WORDS = [
    "this", "shampoo", "keeps", "my", "hair", "soft", "and", "shiny", "for", "days", "and",
    "the", "smell", "is", "light", "fresh", "and", "clean", "so", "i", "will", "buy", "it",
    "again",
]  # fmt: skip


def _write(path: Path, rows: list[tuple[str, float, str, str]]) -> Path:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for i, (text, rating, product, user) in enumerate(rows):
            record = {
                "rating": rating,
                "text": text,
                "parent_asin": product,
                "user_id": user,
                "timestamp": 1_600_000_000_000 + i * 1000,
            }
            handle.write(json.dumps(record) + "\n")
    return path


def test_design_effect_cap() -> None:
    rho = 0.17637632
    cap = design_effect_cap(rho)
    assert cap == 89
    assert information(cap, rho) >= 0.95 / rho > information(cap - 1, rho)
    assert design_effect_cap(0.5) == 19
    with pytest.raises(ValueError):
        design_effect_cap(0.0)


def test_cap_mask_keeps_at_most_cap_per_product() -> None:
    product = np.array([0, 0, 0, 0, 0, 1, 2, 2, 2])
    kept = np.ones(product.size, dtype=bool)
    kept[0] = False  # already removed by an earlier step
    mask = cap_mask(product, kept, cap=2, seed=4213)
    assert np.bincount(product[mask]).tolist() == [2, 1, 2]
    assert not mask[0]
    assert (mask == cap_mask(product, kept, cap=2, seed=4213)).all()  # same seed, same choice


def test_duplicate_rules(tmp_path: Path) -> None:
    long_a = " ".join(WORDS)
    long_b = "Arrived broken and the seller never answered my two emails about a refund at all."
    near_a = " ".join(WORDS + ["truly"])
    near_b = " ".join(WORDS + ["really"])  # same 24 words plus one: overlap 24/26 > 0.9
    rows = [
        (long_a, 5.0, "P1", "A"),
        (long_a, 5.0, "P2", "A"),  # same person, other product: removed
        ("Love it!", 5.0, "P3", "B"),
        ("Love it!", 5.0, "P4", "C"),  # short, another person: kept
        ("Love it!", 5.0, "P5", "B"),  # same person again: removed
        (long_b, 1.0, "P6", "D"),
        (long_b, 1.0, "P7", "E"),  # long, another person: removed
        (near_a, 4.0, "P8", "F"),
        (near_b, 4.0, "P9", "F"),  # near copy by the same person: removed
    ]
    table = build_review_table(_write(tmp_path / "dups.jsonl.gz", rows))
    reasons, original = dedup.duplicate_reasons(table, table.kept_by_step_3())
    assert reasons.tolist() == [
        0, dedup.SAME_PERSON_COPY, 0, 0, dedup.SAME_PERSON_COPY,
        0, dedup.LONG_COPY_OTHER_PERSON, 0, dedup.SAME_PERSON_NEAR_COPY,
    ]  # fmt: skip
    assert original[1] == 0 and original[4] == 2 and original[6] == 5 and original[8] == 7


def test_split_key_is_fixed_and_in_range() -> None:
    assert split_key("B000YB5ZSY") == split_key("B000YB5ZSY")
    assert split_key("B000YB5ZSY") != split_key("B000YB5ZSZ")
    assert 0 <= split_key("anything") < 1


@pytest.fixture()
def pipeline_file(tmp_path: Path) -> Path:
    # Six products whose ratings cluster, so the intraclass correlation is positive.
    # Texts differ by words, not digits: digits all become <num> and would make copies.
    names = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    variants = ["one", "two", "three", "four", "five"]
    rows = []
    for p, stars in enumerate([5, 5, 1, 2, 4, 5]):
        for k, variant in enumerate(variants):
            text = f"The {names[p]} item, review {variant}: " + " ".join(WORDS[k : k + 12])
            rows.append((text, float(stars if k else max(1, stars - 1)), f"P{p}", f"U{p}{k}"))
    rows.append(("", 3.0, "P0", "Z"))  # no words: removed by the length rule
    return _write(tmp_path / "pipeline.jsonl.gz", rows)


def test_pipeline_end_to_end(pipeline_file: Path, tmp_path: Path) -> None:
    prepared = prepare(pipeline_file)
    assert 0 < prepared.rho < 1 and prepared.cap >= 1
    samples = choose_samples(prepared)
    corpus = tmp_path / "out" / "reviews.jsonl.gz"
    excerpts = write_corpus(pipeline_file, prepared, corpus, sample_indices(samples))
    with gzip.open(corpus, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    # Ratings here cluster strongly by product, so the cap bites: each of the six
    # products keeps min(5, cap) of its five reviews.
    assert prepared.cap == design_effect_cap(prepared.rho)
    assert len(rows) == prepared.final.sum() == 6 * min(5, prepared.cap)
    assert {"id", "product", "user", "time", "stars", "split_key", "text"} <= rows[0].keys()
    assert rows[0]["text"].startswith("the alpha item , review ")
    summary = summarize(prepared)
    assert summary["stages"][0]["reviews"] == 31 and summary["final"]["reviews"] == len(rows)
    assert summary["stages"][-2]["reviews"] == 30  # nothing removed before the cap but ""
    assert summary["lengths"]["removed_no_words"] == 1
    assert "Final corpus" in "\n".join(format_report(summary))
    render_samples(prepared, samples, excerpts)


def test_no_cap_when_ratings_of_a_product_are_not_alike(tmp_path: Path) -> None:
    names = ["alpha", "bravo", "charlie"]
    variants = ["one", "two", "three", "four"]
    rows = [
        (
            f"The {name} item, review {variant}: " + " ".join(WORDS[k : k + 12]),
            star,
            name,
            name + variant,
        )
        for name in names
        for k, (variant, star) in enumerate(zip(variants, (1.0, 5.0, 1.0, 5.0), strict=True))
    ]
    prepared = prepare(_write(tmp_path / "flat.jsonl.gz", rows))
    assert prepared.rho is not None and prepared.rho <= 0
    assert prepared.cap is None and prepared.final.sum() == 12
    assert "Cap: none" in "\n".join(format_report(summarize(prepared)))
