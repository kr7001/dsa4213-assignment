"""Tests for the language, product and duplicate checks on a small hand-made file."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from a1.data.duplicate_checks import CATEGORIES, summarize_duplicates
from a1.data.language import sentences_for_language_id, text_for_language_id
from a1.data.language_checks import summarize_language
from a1.data.language_filter import (
    KEEP,
    LONG_FOREIGN,
    MIXED,
    SHORT_FOREIGN,
    english_only_reasons,
)
from a1.data.product_checks import intraclass_correlation, summarize_products
from a1.data.review_table import build_review_table, fetch_excerpts

ENGLISH = [
    "This lotion is good and my skin feels soft every single day.",
    "Good brush, the bristles are soft and it cleans my face well.",
    "I bought this serum last month and it is really good for dry skin.",
    "The color is good but the bottle leaked a little in the box.",
    "Good value for the money, my daughter uses it every morning.",
    "This nail polish dries fast and the color is good on my nails.",
    "Works well on my curly hair and the smell is good and light.",
]
SPANISH = [
    "Me encanta este producto, es muy excelente y suave para el cabello.",
    "Me encanta este producto, es muy excelente y llegó rápido a mi casa.",
    "Me encanta este producto, es muy excelente y huele delicioso.",
    "Me encanta este producto, es muy excelente y mi hija lo usa diario.",
    "Me encanta este producto, es muy excelente y deja la piel brillante.",
    "Me encanta este producto, es muy excelente y lo volvería a comprar.",
]
CROSS_POSTED = "I use this cream every night before bed and my skin looks brighter now."
NEAR_A = "This conditioner keeps my hair soft and easy to brush after every single wash."
NEAR_B = "This conditioner keeps my hair soft and easy to comb after every single wash."
MIXED_REVIEW = (
    "This cream is great and my skin feels soft every single day, I use it every morning "
    "and night and it never feels greasy at all.<br />Me encanta este producto, es muy excelente."
)


def _records() -> list[dict]:
    rows: list[tuple[str, float, str, str]] = []
    rows += [(text, 5.0, f"E{i}", f"UE{i}") for i, text in enumerate(ENGLISH)]
    rows += [(text, 5.0, f"S{i}", f"US{i}") for i, text in enumerate(SPANISH)]
    rows += [("Excelente", 5.0, "S0", "UX1"), ("Good", 4.0, "E0", "UX2")]
    rows += [("Love it!", 5.0, f"P{i}", f"UL{i}") for i in range(3)]  # stock phrase
    rows += [(CROSS_POSTED, 5.0, "C1", "UC"), (CROSS_POSTED, 5.0, "C2", "UC")]  # one user
    rows += [(NEAR_A, 4.0, "N1", "UN"), (NEAR_B, 4.0, "N2", "UN")]  # near copy, one user
    rows += [("", 1.0, "Z", "UZ"), ("10/10", 5.0, "Z", "UZ2")]  # no words
    rows += [("Great item", 5.0, "G1", "UG1"), (MIXED_REVIEW, 5.0, "B1", "UB1")]
    return [
        {"rating": r, "text": t, "parent_asin": p, "user_id": u, "timestamp": 1_600_000_000_000 + i}
        for i, (t, r, p, u) in enumerate(rows)
    ]


@pytest.fixture(scope="module")
def review_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data") / "reviews.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in _records():
            handle.write(json.dumps(record) + "\n")
    return path


@pytest.fixture(scope="module")
def table(review_file: Path):
    return build_review_table(review_file)


def test_table_counts_and_step_3_mask(table) -> None:
    assert len(table) == len(_records())
    kept = table.kept_by_step_3()
    assert kept.sum() == len(table) - 2  # the empty review and "10/10" have no words
    spanish = [table.languages[table.language[i]] for i in range(7, 13)]
    assert spanish == ["es"] * 6


def test_word_evidence_learns_from_confident_long_reviews(table) -> None:
    evidence = table.evidence
    assert evidence.other_reviews == 6
    assert evidence.word_score("excelente") > 2  # in all six Spanish reviews, no English one
    assert evidence.word_score("good") < -2  # in the English reviews only
    assert evidence.word_score("unseenword") == 0.0  # no evidence stays neutral


def test_language_summary_flags_short_spanish(table) -> None:
    kept = table.kept_by_step_3()
    summary, samples = summarize_language(table, kept, seed=4213)
    assert summary["not_english"] >= 6
    foreign = samples["Most common short texts with word evidence 2+"]
    assert any("Excelente" in line for line in foreign)
    assert not any("| Good" in line for line in foreign)


def test_intraclass_correlation() -> None:
    groups = np.array([0, 0, 1, 1])
    assert intraclass_correlation(np.array([1, 1, 5, 5]), groups) == pytest.approx(1.0)
    assert intraclass_correlation(np.array([1, 5, 1, 5]), groups) == pytest.approx(-1.0)
    assert intraclass_correlation(np.array([1, 2]), np.array([0, 1])) is None  # no pairs


def test_product_summary(table) -> None:
    summary = summarize_products(table, table.kept_by_step_3(), seed=4213)
    kept_shares = [row["share_reviews_kept"] for row in summary["cap_table"]]
    assert kept_shares == sorted(kept_shares) and kept_shares[-1] == pytest.approx(1.0)
    assert [row["cap"] for row in summary["split_simulation"]["by_cap"]] == ["none", 30, 100, 200]


def test_duplicates(table, review_file: Path) -> None:
    summary, samples = summarize_duplicates(table, table.kept_by_step_3(), seed=4213)
    rows = {row["group"]: row for row in summary["exact_copies"]["by_length_group"]}
    short = rows["1-2 words"]
    assert short["texts"] == 1 and short["reviews"] == 3
    assert short["extra_copies"][CATEGORIES[3]] == 2  # other users, other products
    long = rows["7-20 words"]
    assert long["extra_copies"][CATEGORIES[1]] == 1  # same user, other product
    near = summary["near_copies_same_user"]
    assert near["near_copy_pairs"] == 1 and near["pairs_other_products"] == 1
    excerpts = fetch_excerpts(review_file, samples.indices())
    rendered = samples.render(excerpts)
    assert "comb" in "\n".join(rendered["Random near copies by the same user (pairs)"])


def test_text_preparation_straightens_quotes_and_splits_sentences() -> None:
    assert text_for_language_id("It\u2019s <b>great</b>") == "it's great"
    raw = "Love it.<br />Me encanta mucho este producto"
    assert sentences_for_language_id(raw, 2) == ["love it", "me encanta mucho este producto"]
    assert sentences_for_language_id(raw, 3) == ["me encanta mucho este producto"]


def test_english_only_rule(table) -> None:
    reasons = english_only_reasons(table)
    by_text = {record["text"]: reasons[i] for i, record in enumerate(_records())}
    assert by_text["Excelente"] == SHORT_FOREIGN  # rule 1: its word is typical of Spanish
    assert all(by_text[text] == LONG_FOREIGN for text in SPANISH)  # rule 2
    assert by_text[MIXED_REVIEW] == MIXED  # rule 3: English with a Spanish sentence
    assert by_text["Great item"] == KEEP  # fastText says Catalan; its words say otherwise
    assert all(by_text[text] == KEEP for text in ENGLISH)
