"""Tests for the vocabulary measurements."""

from __future__ import annotations

import gzip
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from a1.data.vocabulary_checks import (
    count_halves,
    format_vocabulary,
    frequency_order,
    glove_coverage,
    min_count_rows,
    read_glove_words,
    split_like_glove,
    summarize_vocabulary,
    unknown_rates,
    worst_ranks,
)


def _corpus(path: Path, rows: list[tuple[float, str]]) -> Path:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for i, (key, text) in enumerate(rows):
            row = {"id": i, "product": f"P{i}", "user": f"U{i}", "time": i, "stars": 5}
            handle.write(json.dumps(row | {"split_key": key, "text": text}) + "\n")
    return path


def test_unknown_rates_count_held_out_tokens_and_reviews(tmp_path: Path) -> None:
    path = _corpus(
        tmp_path / "c.jsonl.gz",
        [
            (0.1, "love it ."),
            (0.2, "love it so much ."),
            (0.3, "don't buy ."),
            (0.7, "love it ."),  # all known to half A
            (0.8, "hate it ."),  # "hate" never seen in half A
        ],
    )
    counts, reviews = count_halves(path)
    assert reviews == [3, 2] and counts[0]["love"] == 2 and counts[1]["hate"] == 1
    orders = [frequency_order(c) for c in counts]
    assert orders[0][:3] == [".", "it", "love"]  # most frequent first, ties alphabetical
    worst = worst_ranks(path, orders)
    rows = unknown_rates(counts[0], counts[1], worst[1], sizes=(3,))
    # A vocabulary of the 3 most frequent tokens of half A covers ". it love" but
    # not "hate": 1 of the 6 held-out tokens, in 1 of the 2 held-out reviews.
    first = rows[0]
    assert first["vocabulary_size"] == 7 and first["smallest_count_kept"] == 2
    assert np.isclose(first["held_out_unk_share"], 1 / 6)
    assert first["held_out_reviews_with_unk"] == 0.5
    assert rows[-1]["kept_tokens"] == len(orders[0])  # the full vocabulary is always added
    by_count = min_count_rows(counts[0], counts[1], minimums=(1, 2))
    assert [row["vocabulary_size"] for row in by_count] == [11, 7]


def test_split_like_glove() -> None:
    assert split_like_glove("don't") == ["do", "n't"]
    assert split_like_glove("can't") == ["ca", "n't"]
    assert split_like_glove("won't") == ["wo", "n't"]
    assert split_like_glove("it's") == ["it", "'s"]
    assert split_like_glove("i'm") == ["i", "'m"]
    assert split_like_glove("they're") == ["they", "'re"]
    assert split_like_glove("y'all") == ["y'all"]
    assert split_like_glove("great") == ["great"]


def test_read_glove_words_in_three_formats(tmp_path: Path) -> None:
    lines = ["the 0.1 0.2", "do 0.3 0.4", "n't 0.5 0.6"]
    txt = tmp_path / "g.txt"
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    gz = tmp_path / "g.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as handle:
        handle.write("3 2\n" + "\n".join(lines) + "\n")  # gensim header first
    zipped = tmp_path / "g.zip"
    with zipfile.ZipFile(zipped, "w") as archive:
        archive.writestr("vectors.txt", "\n".join(lines) + "\n")
    expected = {"the", "do", "n't"}
    assert read_glove_words(txt) == read_glove_words(gz) == read_glove_words(zipped) == expected


def test_glove_coverage_whole_and_split() -> None:
    build = Counter({"i": 5, "don't": 4, "love": 3, "<num>": 2})
    held = Counter({"i": 2, "don't": 2, "love": 1, "<num>": 1})
    glove = {"i", "do", "n't", "love"}
    row = glove_coverage(build, held, glove, sizes=(4,))[0]
    assert row["types_covered_whole"] == 0.5  # "i" and "love"
    assert row["types_covered_split"] == 0.75  # plus "don't" as "do" + "n't"
    assert np.isclose(row["token_share_covered_whole"], 3 / 6)
    assert np.isclose(row["token_share_covered_split"], 5 / 6)
    assert row["contraction_types_covered_whole"] == 0
    assert row["contraction_types_covered_split"] == 1
    assert row["most_frequent_missing_after_split"] == {"<num>": 2}


def test_summary_end_to_end(tmp_path: Path) -> None:
    path = _corpus(
        tmp_path / "c.jsonl.gz",
        [(0.1, "i don't love it ."), (0.2, "love it ."), (0.6, "i love it !"), (0.9, "it's ok .")],
    )
    glove = tmp_path / "g.txt"
    glove.write_text("i 0\nlove 0\nit 0\n. 0\ndo 0\nn't 0\n", encoding="utf-8")
    summary = summarize_vocabulary(path, [glove])
    assert summary["halves"]["reviews"] == [2, 2]
    assert summary["contractions"]["types"] == 2
    assert "g.txt" in summary["glove"]
    assert summary["chosen_rule"]["min_count"] == 5
    assert summary["chosen_rule"]["kept_tokens_on_a"] == 0  # nothing is seen 5 times here
    assert "Contractions:" in "\n".join(format_vocabulary(summary))
