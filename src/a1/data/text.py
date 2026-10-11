"""Text helpers shared by every stage of the pipeline.

The tokenizer is the Lecture 1 course tokenizer, adapted from DSA4213
Lecture1_Demo.ipynb (D. Zhou, 2026): lowercase the text, turn the curly
apostrophe into a straight one, then keep words (with an optional apostrophe
part, so "don't" stays whole) and six punctuation marks. Everything else,
including digits, other symbols and emoji, is dropped.

The noise patterns below only measure what the raw reviews contain. Which
cleaning steps to apply is decided later, from those measurements.
"""

from __future__ import annotations

import re

COURSE_TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|[.,!?;:]")

# Each pattern flags one kind of raw-text artefact that could need cleaning.
NOISE_PATTERNS: dict[str, re.Pattern[str]] = {
    # A tag is "<", an optional "/", then a letter, e.g. "<br />", "<p>", "</a>".
    # Requiring a letter keeps emoticons such as "<3" out.
    "html_tag": re.compile(r"<\s*/?\s*[a-zA-Z][a-zA-Z0-9]*(?:\s[^<>]{0,200})?\s*/?>"),
    # Escaped characters such as "&amp;", "&#39;" or "&#x27;". The course regex
    # would turn the ";" of each one into a spurious punctuation token.
    "html_entity": re.compile(r"&(?:[a-zA-Z][a-zA-Z0-9]{1,7}|#[0-9]{1,6}|#[xX][0-9a-fA-F]{1,6});"),
    "url": re.compile(r"https?://|\bwww\.", re.IGNORECASE),
    # Curly quotes. The course tokenizer already maps the right single quote
    # (the usual apostrophe) to "'"; the others are simply dropped.
    "curly_quote": re.compile("[‘’“”]"),
    "emoji": re.compile("[\U0001f300-\U0001faff☀-➿]"),
    "digit": re.compile(r"[0-9]"),
}

# Very common English words. A review of several tokens containing none of
# them is worth a look as possibly not English (the PDF requires English).
COMMON_ENGLISH_WORDS = frozenset(
    {
        "the", "and", "a", "an", "i", "it", "is", "to", "this", "for", "of", "my", "in",
        "was", "not", "but", "with", "that", "you", "on", "so", "are", "have", "be",
        "very", "as", "they", "its", "it's",
    }
)  # fmt: skip


def course_tokenize(text: str) -> list[str]:
    """Split raw text into tokens with the Lecture 1 course tokenizer.

    Args:
        text: Raw review text in any case.

    Returns:
        Lowercase tokens in their original order: words, with contractions such
        as "don't" kept whole, and the marks . , ! ? ; : as separate tokens.
    """
    normalized = text.lower().replace("’", "'")
    return COURSE_TOKEN_PATTERN.findall(normalized)


def non_ascii_letter_share(text: str) -> float:
    """Return the share of letters in ``text`` that fall outside plain ASCII.

    A high share suggests a non-Latin script, for example a review in Chinese
    or Arabic. Accented Latin letters (as in Spanish) count as non-ASCII too.

    Args:
        text: Raw review text.

    Returns:
        A value in [0, 1]; 0 when the text has no letters.
    """
    if text.isascii():
        return 0.0
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for char in letters if not char.isascii()) / len(letters)
