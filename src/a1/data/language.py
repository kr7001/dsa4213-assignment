"""Language identification, used to keep the corpus English only.

Two kinds of evidence:

1. fastText's language identifier (lid.176, Joulin et al. 2016, 2017), the
   compressed version that ships inside the fast-langdetect package, so no
   download is needed and every run gives the same answer. It reads character
   pieces of words and works well on longer text.
2. Word evidence for short reviews, where a few words give fastText too little
   to go on. Reviews of 7+ words that fastText labels with high confidence
   show which words are typical of English and which of other languages; a
   short review is then judged by its words.

Text is lowercased before identification: the identifier is case-sensitive,
and capitalised single words ("Buena", "Good") otherwise look like names.
Curly quotes become straight ones: on our data the curly apostrophe in "It’s"
alone pushed fastText to guess Russian or Thai for plainly English reviews.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from fast_langdetect import LangDetectConfig, LangDetector

from a1.data.clean import strip_markup

ENGLISH = "en"

# The English-only rule (decision D0.9); see language_filter.py.
SHORT_MAX_WORDS = 6  # up to here a review is judged by its words, not by fastText
SHORT_MIN_EVIDENCE = 2.0  # 1-6 words: removed if the word score reaches this
LONG_SURE_PROBABILITY = 0.9  # 7+ words: fastText alone decides at this confidence
LONG_MIN_EVIDENCE = 2.0  # 7+ words: below it, the word score has to agree
SENTENCE_MIN_WORDS = 5  # sentences checked one by one inside reviews of 7+ words
SENTENCE_MIN_PROBABILITY = 0.9
SENTENCE_MIN_MEAN_EVIDENCE = 2.0  # average word score of a foreign sentence
# Letters in any script, so accented words such as "llegó" stay whole.
_UNICODE_WORD = re.compile(r"[^\W\d_]+")
_STRAIGHT_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})
_LINE_BREAK = re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE)
# Sentence ends, line breaks and slashes: a translation often follows " / ".
_SENTENCE_END = re.compile(r"[.!?\n/]+")


def text_for_language_id(raw: str) -> str:
    """Prepare raw review text for language identification.

    Removes tags, links and hearts and decodes escaped characters, like the
    step 1 cleaning, but keeps accents, digits and symbols, which carry
    language evidence. Straightens curly quotes, lowercases, and puts
    everything on one line.

    Args:
        raw: Raw review text.

    Returns:
        One lowercase line of text.
    """
    return " ".join(strip_markup(raw).translate(_STRAIGHT_QUOTES).split()).lower()


def sentences_for_language_id(raw: str, min_words: int) -> list[str]:
    """Split a review into sentences prepared like :func:`text_for_language_id`.

    Line breaks ("<br />") and slashes also end a sentence, since a
    translation often starts on a new line or after " / ".

    Args:
        raw: Raw review text.
        min_words: Sentences with fewer words (in any script) are left out.

    Returns:
        The prepared sentences, in order.
    """
    text = strip_markup(_LINE_BREAK.sub("\n", raw)).translate(_STRAIGHT_QUOTES).lower()
    pieces = (" ".join(piece.split()) for piece in _SENTENCE_END.split(text))
    return [piece for piece in pieces if len(unicode_words(piece)) >= min_words]


def unicode_words(text: str) -> list[str]:
    """Return the runs of letters in ``text``, in any script."""
    return _UNICODE_WORD.findall(text)


@dataclass(frozen=True)
class LanguageGuess:
    """The identifier's two most likely languages and their probabilities."""

    language: str
    probability: float
    runner_up: str
    runner_up_probability: float

    @property
    def is_english(self) -> bool:
        return self.language == ENGLISH


class LanguageIdentifier:
    """fastText lid.176 (compressed model bundled with fast-langdetect)."""

    def __init__(self) -> None:
        config = LangDetectConfig(max_input_length=None, normalize_input=False, model="lite")
        self._detector = LangDetector(config)

    def guess(self, text: str) -> LanguageGuess:
        """Identify the language of text prepared by :func:`text_for_language_id`."""
        results = self._detector.detect(text, model="lite", k=2)
        first = results[0]
        second = results[1] if len(results) > 1 else {"lang": "", "score": 0.0}
        return LanguageGuess(
            language=str(first["lang"]),
            probability=float(first["score"]),
            runner_up=str(second["lang"]),
            runner_up_probability=float(second["score"]),
        )


@dataclass
class WordEvidence:
    """How typical each word is of non-English versus English reviews.

    Fit on reviews whose language is known with confidence. Each word scores
    the log of how much more often it appears in non-English reviews than in
    English ones (share of reviews containing it), so positive means foreign
    and negative means English. Words seen fewer than ``min_reviews`` times in
    total score 0: there is not enough evidence, so typos and rare names stay
    neutral instead of looking foreign.
    """

    min_reviews: int = 5
    max_score: float = 5.0
    english_reviews: int = 0
    other_reviews: int = 0
    english_counts: Counter[str] = field(default_factory=Counter)
    other_counts: Counter[str] = field(default_factory=Counter)

    def add(self, words: Iterable[str], is_english: bool) -> None:
        """Count each distinct word of one confidently identified review."""
        distinct = set(words)
        if is_english:
            self.english_reviews += 1
            self.english_counts.update(distinct)
        else:
            self.other_reviews += 1
            self.other_counts.update(distinct)

    def word_score(self, word: str) -> float:
        english, other = self.english_counts[word], self.other_counts[word]
        if english + other < self.min_reviews or not self.english_reviews or not self.other_reviews:
            return 0.0
        # Half a review of smoothing keeps words absent from one side finite.
        other_share = (other + 0.5) / (self.other_reviews + 1)
        english_share = (english + 0.5) / (self.english_reviews + 1)
        score = math.log(other_share / english_share)
        return max(-self.max_score, min(self.max_score, score))

    def score(self, words: Iterable[str]) -> float:
        """Sum of word scores: above 0 leans foreign, below 0 leans English."""
        return sum(self.word_score(word) for word in words)

    def mean_score(self, words: list[str]) -> float:
        """Average word score, so a long sentence and a short one compare fairly."""
        return self.score(words) / len(words) if words else 0.0
