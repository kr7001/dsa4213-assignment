"""Cleaning applied to every review before it is tokenized (step 1 decisions).

Each rule comes from the step 1 measurements in results/eda/summary.json:

- HTML tags such as "<br />", Amazon's video tags such as
  "[[VIDEOID:2f73...]]" and web links become a space.
- Amazon's links to another product, "[[ASIN:B000YB5ZSY Product Name]]",
  keep the product name and lose the code, since the name is part of the
  sentence.
- Escaped characters are decoded: "&#39;" becomes "'" and "&amp;" becomes "&".
- "&" becomes the word "and", so "Tom & Jerry" keeps its meaning.
- The heart "<3" (and the broken heart "</3") is dropped like other emoji,
  so its 3 does not turn into a number.
- Every number becomes the single token "<num>": "2.5 oz" becomes "<num> oz".

The tokenizer is the course tokenizer from text.py with one addition: it keeps
"<num>" as one token. Emoji and other symbols are still dropped, which is a
stated limitation.
"""

from __future__ import annotations

import html
import re

from a1.data.text import NOISE_PATTERNS

NUM_TOKEN = "<num>"

_HTML_TAG = NOISE_PATTERNS["html_tag"]
# Amazon marks an attached video with "[[VIDEOID:" and a hex code; the code is not language.
VIDEO_TAG = re.compile(r"\[\[VIDEOID:[^\]]*\]\]", re.IGNORECASE)
# Amazon links another product as "[[ASIN:<code> <product name>]]".
ASIN_TAG = re.compile(r"\[\[ASIN:\s*[A-Za-z0-9]*\s*([^\]]*)\]\]", re.IGNORECASE)
# A web link runs from "http(s)://" or "www." up to the next space.
_URL = re.compile(r"(?:https?://|\bwww\.)\S*", re.IGNORECASE)
# "<3" is a heart and "</3" a broken heart, sometimes written "<333".
_HEART = re.compile(r"</?3+")
# Digits with optional inner separators, so "1,000" and "2.5" are one number each.
_NUMBER = re.compile(r"[0-9]+(?:[.,][0-9]+)*")
TOKEN_PATTERN = re.compile(r"<num>|[a-z]+(?:'[a-z]+)?|[.,!?;:]")


def strip_markup(text: str) -> str:
    """Remove markup, links and hearts, and decode escaped characters.

    Tags are removed before escaped characters are decoded, so an escaped
    "&lt;3" cannot turn into a fake tag that swallows nearby words. Hearts are
    removed after decoding, so both "<3" and "&lt;3" disappear.

    Args:
        text: Raw review text.

    Returns:
        The text without markup, in its original case, with accents, digits
        and symbols kept.
    """
    text = ASIN_TAG.sub(lambda match: f" {match.group(1)} ", text)
    text = _HTML_TAG.sub(" ", text)
    text = VIDEO_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = html.unescape(text)
    return _HEART.sub(" ", text)


def clean_text(text: str) -> str:
    """Apply the step 1 cleaning rules to one raw review.

    Args:
        text: Raw review text.

    Returns:
        The cleaned text, still in its original case.
    """
    text = strip_markup(text).replace("&", " and ")
    return _NUMBER.sub(f" {NUM_TOKEN} ", text)


def tokenize(text: str) -> list[str]:
    """Split cleaned text with the course tokenizer, keeping "<num>" whole.

    Args:
        text: Output of :func:`clean_text`.

    Returns:
        Lowercase tokens: words (contractions kept whole), "<num>", and the
        marks . , ! ? ; : as separate tokens.
    """
    normalized = text.lower().replace("’", "'")
    return TOKEN_PATTERN.findall(normalized)


def is_word(token: str) -> bool:
    """Return True for a word token, False for punctuation and "<num>"."""
    return token[0].isalpha()


def count_words(tokens: list[str]) -> int:
    """Count the word tokens, leaving out punctuation and "<num>"."""
    return sum(1 for token in tokens if token[0].isalpha())
