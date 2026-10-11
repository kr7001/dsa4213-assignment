"""Tests for the course tokenizer and the raw-text noise checks."""

from a1.data.text import NOISE_PATTERNS, course_tokenize, non_ascii_letter_share


def test_course_tokenizer_matches_lecture_examples() -> None:
    expected = ["i", "love", "data", ",", "statistics", ",", "and", "nlp", "!"]
    assert course_tokenize("I love data, statistics, and NLP!") == expected


def test_contractions_stay_whole_and_curly_apostrophe_is_normalized() -> None:
    assert course_tokenize("I don't like it") == ["i", "don't", "like", "it"]
    assert course_tokenize("I don’t like it") == ["i", "don't", "like", "it"]


def test_digits_and_other_symbols_are_dropped() -> None:
    assert course_tokenize("Only $12.99 (2 pack) :)") == ["only", ".", "pack", ":"]


def test_html_entity_leaves_a_spurious_semicolon_token() -> None:
    # Why the entity check matters: "&#39;" is an escaped apostrophe.
    assert course_tokenize("don&#39;t") == ["don", ";", "t"]


def test_noise_patterns_flag_what_they_should() -> None:
    def flagged(text: str) -> set[str]:
        return {name for name, pattern in NOISE_PATTERNS.items() if pattern.search(text)}

    assert flagged("Great<br />Would buy again") == {"html_tag"}
    assert flagged("I <3 this") == {"digit"}  # an emoticon, not a tag
    assert flagged("Tom &amp; Jerry") == {"html_entity"}
    assert flagged("see https://example.com") == {"url"}
    assert flagged("it’s nice") == {"curly_quote"}
    assert flagged("love it \U0001f60d") == {"emoji"}


def test_non_ascii_letter_share() -> None:
    assert non_ascii_letter_share("plain text") == 0.0
    assert non_ascii_letter_share("非常好") == 1.0
    assert non_ascii_letter_share("") == 0.0
