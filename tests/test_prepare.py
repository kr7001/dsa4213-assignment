"""Tests for the cap, the duplicate rules and the whole data pipeline."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from a1.data import dedup
from a1.data.cap import cap_mask, cap_on_random_halves, design_effect_cap, information
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
    # A change in one product never reshuffles the reviews another product keeps.
    kept_more = kept.copy()
    kept_more[0] = True
    other = cap_mask(product, kept_more, cap=2, seed=4213)
    assert (other[6:] == mask[6:]).all()


def test_cap_on_random_halves_is_stable_and_seeded() -> None:
    rng = np.random.default_rng(0)
    products = np.repeat(np.arange(200), 20)
    ratings = np.clip(
        np.round(rng.normal(3, 1, 200)[products] + rng.normal(0, 1, products.size)), 1, 5
    )
    check = cap_on_random_halves(ratings, products, seed=4213)
    assert check["repeats"] == 50 and check["no_cap"] == 0
    assert check["cap"]["min"] <= check["cap"]["median"] <= check["cap"]["max"]
    assert check == cap_on_random_halves(ratings, products, seed=4213)


def test_duplicate_rules(tmp_path: Path) -> None:
    long_a = " ".join(WORDS)
    long_b = "Arrived broken and the seller never answered my two emails about a refund at all."
    near_a = " ".join(WORDS + ["truly"])
    near_b = " ".join(WORDS + ["really"])  # same 24 words plus one: overlap 24/26 > 0.9
    words_13 = "I bought this as a gift for my husband and he loves it"
    words_14 = "I bought this as a gift for my husband and he really loves it"
    rows = [
        (long_a, 5.0, "P1", "A"),
        (long_a, 5.0, "P2", "A"),  # same person, other product: removed
        ("Love it!", 5.0, "P3", "B"),
        ("Love it!", 5.0, "P4", "C"),  # short, another person: kept
        ("Love it!", 5.0, "P5", "B"),  # same person again: removed
        (long_b, 1.0, "P6", "D"),
        (long_b, 1.0, "P7", "E"),  # 15 words, another person: removed
        (near_a, 4.0, "P8", "F"),
        (near_b, 4.0, "P9", "F"),  # near copy by the same person: removed
        (words_13, 5.0, "P10", "G"),
        (words_13, 5.0, "P11", "H"),  # 13 words, another person: kept
        (words_14, 5.0, "P12", "I"),
        (words_14, 5.0, "P13", "J"),  # 14 words, another person: removed
    ]
    table = build_review_table(_write(tmp_path / "dups.jsonl.gz", rows))
    assert table.words[9] == 13 and table.words[11] == dedup.OTHER_PERSON_MIN_WORDS == 14
    reasons, original = dedup.duplicate_reasons(table, table.kept_by_step_3())
    assert reasons.tolist() == [
        0, dedup.SAME_PERSON_COPY, 0, 0, dedup.SAME_PERSON_COPY,
        0, dedup.OTHER_PERSON_COPY, 0, dedup.SAME_PERSON_NEAR_COPY,
        0, 0, 0, dedup.OTHER_PERSON_COPY,
    ]  # fmt: skip
    # Every repeat points at the earliest review with its text, kept or removed.
    assert original.tolist() == [-1, 0, -1, 2, 2, -1, 5, -1, 7, -1, 9, -1, 11]


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
    rows[7] = (rows[7][0] + " Ask me at bob.smith@gmail.com", *rows[7][1:])  # removed email
    rows.append(("", 3.0, "P0", "Z"))  # no words: removed by the length rule
    return _write(tmp_path / "pipeline.jsonl.gz", rows)


def test_pipeline_end_to_end(pipeline_file: Path, tmp_path: Path) -> None:
    prepared = prepare(pipeline_file)
    assert 0 < prepared.rho < 1 and prepared.cap >= 1
    samples = choose_samples(prepared)
    corpus = tmp_path / "out" / "reviews.jsonl.gz"
    written = write_corpus(pipeline_file, prepared, corpus, sample_indices(samples))
    with gzip.open(corpus, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    # The fingerprint is the SHA-256 of the uncompressed file, and the same data
    # always gives the same bytes on disk.
    assert written.rows == len(rows)
    assert written.sha256 == hashlib.sha256(gzip.decompress(corpus.read_bytes())).hexdigest()
    again = tmp_path / "again" / "reviews.jsonl.gz"
    write_corpus(pipeline_file, prepared, again, set())
    assert again.read_bytes() == corpus.read_bytes()
    # Ratings here cluster strongly by product, so the cap bites: each of the six
    # products keeps min(5, cap) of its five reviews.
    assert prepared.cap == design_effect_cap(prepared.rho)
    assert len(rows) == prepared.final.sum() == 6 * min(5, prepared.cap)
    assert {"id", "product", "user", "time", "stars", "split_key", "text"} <= rows[0].keys()
    assert rows[0]["text"].startswith("the alpha item , review ")
    summary = summarize(prepared, written)
    assert summary["stages"][0]["reviews"] == 31 and summary["final"]["reviews"] == len(rows)
    assert summary["stages"][-2]["reviews"] == 30  # nothing removed before the cap but ""
    assert summary["lengths"]["removed_no_words"] == 1
    assert summary["corpus"] == {"rows": len(rows), "sha256_uncompressed": written.sha256}
    assert summary["email_addresses_removed_from_reviews"] == 1
    assert not any("gmail" in row["text"] for row in rows)
    assert summary["cap"]["check_random_halves"]["repeats"] == 50
    report = "\n".join(format_report(summary))
    assert "Final corpus" in report and written.sha256 in report
    rendered = render_samples(prepared, samples, written.excerpts)
    assert any(title.startswith("Duplicates, kept: exact copy of 7-13 words") for title in rendered)


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
    assert prepared.cap_check is None
    assert "Cap: none" in "\n".join(format_report(summarize(prepared)))
