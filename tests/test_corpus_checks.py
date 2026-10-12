"""Tests for the checks on the clean corpus."""

from __future__ import annotations

import gzip
import hashlib
import itertools
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from a1.data.corpus_checks import (
    Corpus,
    exact_repeats,
    format_checks,
    leftovers,
    load_corpus,
    near_identical_pairs,
    run_checks,
    summarize_near_pairs,
)

WORDS_31 = (
    "this serum soaks in fast and leaves my dry skin soft for hours without any greasy "
    "film so i use it every night before bed and my face looks brighter now"
)


def _corpus_file(path: Path, rows: list[tuple[str, str, str]]) -> Path:
    """Write (text, product, user) rows in the clean corpus format."""
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for i, (text, product, user) in enumerate(rows):
            row = {
                "id": i,
                "product": product,
                "user": user,
                "time": 1_600_000_000_000 + i,
                "stars": 5,
                "split_key": 0.5,
                "text": text,
            }
            handle.write(json.dumps(row) + "\n")
    return path


def _corpus(texts: list[str], users: list[int], products: list[int] | None = None) -> Corpus:
    n = len(texts)
    products = products if products is not None else list(range(n))
    return Corpus(
        ids=np.arange(n),
        product=np.array(products),
        user=np.array(users),
        time=np.arange(n),
        stars=np.full(n, 5),
        split_key=np.full(n, 0.5),
        text=texts,
        product_names=[str(p) for p in range(max(products) + 1)],
        user_names=[str(u) for u in range(max(users) + 1)],
        schemas=Counter(),
        sha256="",
    )


def test_checks_on_a_small_corpus(tmp_path: Path) -> None:
    gift = "i bought this as a gift for my husband and he loves it"  # 13 words
    rows = [
        (gift + " !", "P1", "A"),
        (gift + " .", "P2", "B"),  # same words, other person: near-identical, 13 words
        (WORDS_31, "P3", "C"),
        (WORDS_31.replace("brighter", "clearer"), "P3", "D"),  # one word changed: 31 words
        ("just what i needed at a great price", "P4", "E"),
        ("just what i needed at a great price", "P5", "F"),  # exact copy, kept below 14
        ("don't buy , it s bad ! ! ! ! !", "P6", "G"),  # one whole, one split, a run of 5
        ("mail me at jane <num> gmail . com", "P7", "H"),  # an email address
        ("the cr me feels rich", "P8", "I"),  # a broken accented word
        ("el producto es muy bueno para mi piel", "P9", "J"),  # Spanish function words
    ]
    path = _corpus_file(tmp_path / "corpus.jsonl.gz", rows)
    corpus = load_corpus(path)
    assert corpus.sha256 == hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest()
    tokens = sum(len(text.split()) for text, _, _ in rows)
    report = {
        "cap": {"cap": 90},
        "final": {"reviews": len(rows), "tokens": tokens},
        "corpus": {"rows": len(rows), "sha256_uncompressed": corpus.sha256},
    }
    results, pairs = run_checks(corpus, report)

    structure = results["structure"]
    assert structure["matches_report"] == {"rows": True, "tokens": True, "fingerprint": True}
    assert structure["invalid_tokens"] == 0 and structure["within_length_limits"]
    assert structure["products_with_several_split_keys"] == 0

    exact = results["duplicates"]["exact"]
    assert exact["same_person_same_text"] == 0
    assert exact["copies_by_other_people_kept"] == {"8": 1}
    near = results["duplicates"]["near_identical"]
    assert len(WORDS_31.split()) == 31
    assert near["by_words_of_longer_review"] == {"13": 1, "31": 1}
    assert near["longest_pair_below_14_words"] == 13
    assert near["shortest_pair_of_14_plus_words"] == 31
    assert near["pairs_of_14_plus_words_on_the_same_product"] == 1

    left = results["leftovers"]
    assert left["contractions"]["whole"] == 1 and left["contractions"]["split"] == 1
    assert left["punctuation_runs"]["tokens_in_runs_of_4_plus"] == 5
    assert left["punctuation_runs"]["longest_run"] == 5
    assert left["email_addresses"]["reviews"] == 1
    assert left["accent_fragments"]["reviews_by_word"] == {"crème": 1}
    assert left["reviews_with_4_plus_foreign_function_words"]["es"] == 1
    assert "Same person, same text: 0" in "\n".join(format_checks(results))
    assert len(pairs) == 2


def test_near_identical_search_is_exact() -> None:
    # Random texts over a small vocabulary, plus planted near copies, so that many
    # pairs sit around the threshold. The search must find exactly the pairs that
    # a comparison of every pair finds.
    rng = random.Random(4213)
    vocabulary = [f"w{k}" for k in range(40)]
    texts: list[str] = []
    for _ in range(150):
        texts.append(" ".join(rng.sample(vocabulary, rng.randint(7, 25))))
    for k in range(40):  # near copies: drop or swap one word
        words = texts[k].split()
        words[rng.randrange(len(words))] = rng.choice(vocabulary)
        texts.append(" ".join(words))
    texts = list(dict.fromkeys(texts))  # the search compares distinct texts
    corpus = _corpus(texts, users=list(range(len(texts))))
    pairs, complete = near_identical_pairs(corpus)
    found = {frozenset((corpus.text[p.first], corpus.text[p.second])) for p in pairs}
    expected = set()
    for a, b in itertools.combinations(texts, 2):
        x, y = set(a.split()), set(b.split())
        if len(x & y) / len(x | y) >= 0.9:
            expected.add(frozenset((a, b)))
    assert complete and expected and found == expected


def test_near_identical_search_stops_at_its_budget() -> None:
    texts = [" ".join(f"w{(k + i) % 9}" for i in range(8)) + f" x{k}" for k in range(30)]
    corpus = _corpus(texts, users=list(range(len(texts))))
    _, complete = near_identical_pairs(corpus, threshold=0.5, max_comparisons=10)
    assert not complete


def test_exact_repeats_by_length_and_person() -> None:
    long_text = " ".join(f"w{k}" for k in range(14))
    corpus = _corpus(["love it", "love it", "love it", long_text, long_text], users=[0, 1, 0, 2, 3])
    exact = exact_repeats(corpus)
    assert exact["same_person_same_text"] == 1
    assert exact["copies_by_other_people_kept"] == {"1-6": 1}
    assert exact["copies_of_14_plus_words_by_other_people"] == 1


def test_leftovers_count_runs_only_from_four_marks() -> None:
    corpus = _corpus(["wow ! ! ! great", "wow . . . . ok"], users=[0, 1])
    runs = leftovers(corpus)["punctuation_runs"]
    assert runs["tokens_in_runs_of_4_plus"] == 4 and runs["reviews"] == 1


def test_summary_of_pairs_without_any() -> None:
    summary = summarize_near_pairs([])
    assert summary["pairs_by_other_people"] == 0
    assert summary["shortest_pair_of_14_plus_words"] is None
