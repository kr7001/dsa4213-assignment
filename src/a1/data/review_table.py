"""One pass over the raw reviews that keeps what the filter decisions need.

For every review it keeps the product, user, time, stars, length after
cleaning, the id of its exact cleaned text and a language guess. Long reviews
also keep the set of their words, for spotting near copies by the same user.
Short reviews keep their words, for judging their language by word evidence.
"""

from __future__ import annotations

import hashlib
import re
from array import array
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from a1.data.clean import clean_text, has_email, tokenize
from a1.data.explore import iter_reviews
from a1.data.language import (
    SENTENCE_MIN_PROBABILITY,
    SENTENCE_MIN_WORDS,
    SHORT_MAX_WORDS,
    LanguageIdentifier,
    WordEvidence,
    sentences_for_language_id,
    text_for_language_id,
    unicode_words,
)

# Length groups fixed in step 3 (words after cleaning), used for every breakdown.
LENGTH_GROUPS: tuple[tuple[int, int | None], ...] = ((1, 2), (3, 6), (7, 20), (21, None))
CONFIDENT = 0.9  # fastText probability for a long review to teach word evidence
MAX_TOKENS = 512  # step 3: longer reviews are removed
EXCERPT_CHARS = 160
NO_LANGUAGE = ""  # reviews with no words are not identified
# Amazon markup such as "[[VIDEOID:...]]" or "[[ASIN:...]]", counted by its name.
_MARKUP = re.compile(r"\[\[([A-Za-z]+):")
MARKUP_EXAMPLES = 10


def group_name(low: int, high: int | None) -> str:
    return f"{low}+ words" if high is None else f"{low}-{high} words"


def in_group(words: np.ndarray, low: int, high: int | None) -> np.ndarray:
    """True where a review has between ``low`` and ``high`` words (no upper end if None)."""
    inside = words >= low
    return inside if high is None else inside & (words <= high)


def excerpt(raw: str) -> str:
    """Return the review on one line, cut to EXCERPT_CHARS characters."""
    text = " ".join(raw.split())
    return text if len(text) <= EXCERPT_CHARS else text[:EXCERPT_CHARS] + "..."


@dataclass
class ShortReview:
    """What the language check needs from a review of 1 to SHORT_MAX_WORDS words."""

    words: list[str]  # letters in any script, from the text prepared for identification
    text: str  # the prepared text, used to group identical short reviews
    excerpt: str


@dataclass
class ForeignSentence:
    """A sentence of a long review that fastText calls another language with confidence."""

    text: str
    language: str
    probability: float
    words: list[str]


@dataclass
class ReviewTable:
    """Per-review facts, aligned by position in the raw file.

    Attributes:
        product, user: Integer ids (first appearance order).
        timestamp: Milliseconds since 1970.
        rating: Stars, 1 to 5.
        words, tokens: Counts after the step 1 cleaning.
        text_id: Same id for reviews whose cleaned token sequence is identical.
        language, probability: fastText's top guess (index into ``languages``).
        runner_up, runner_up_probability: Its second guess.
        word_ids, word_offsets: For reviews with SHORT_MAX_WORDS + 1 or more
            words, the sorted distinct word ids of review i are
            ``word_ids[word_offsets[i]:word_offsets[i + 1]]`` (empty otherwise).
        short_reviews: Review index to its ShortReview, for 1 to SHORT_MAX_WORDS words.
        flagged_excerpts: Review index to excerpt, for reviews fastText does
            not call English.
        foreign_sentences: For reviews of SHORT_MAX_WORDS + 1 or more words,
            their sentences of SENTENCE_MIN_WORDS+ words that fastText calls
            another language with probability SENTENCE_MIN_PROBABILITY+.
        evidence: Word evidence learnt from confidently identified long reviews.
        markup_reviews: Reviews containing each kind of "[[NAME:" markup.
        markup_examples: A few raw markup strings per kind, as they appear.
        email_reviews: Reviews that had an email address (removed by cleaning).
        flagged_long_words: For reviews of SHORT_MAX_WORDS + 1 or more words
            that fastText does not call English, their words (any script),
            so the word score can check fastText's call.
        product_names, user_names: The original ids, indexed by the integer ids.
    """

    product: np.ndarray
    user: np.ndarray
    timestamp: np.ndarray
    rating: np.ndarray
    words: np.ndarray
    tokens: np.ndarray
    text_id: np.ndarray
    language: np.ndarray
    probability: np.ndarray
    runner_up: np.ndarray
    runner_up_probability: np.ndarray
    languages: list[str]
    word_ids: np.ndarray
    word_offsets: np.ndarray
    short_reviews: dict[int, ShortReview]
    flagged_excerpts: dict[int, str]
    evidence: WordEvidence
    foreign_sentences: dict[int, list[ForeignSentence]] = field(default_factory=dict)
    markup_reviews: Counter[str] = field(default_factory=Counter)
    markup_examples: dict[str, list[str]] = field(default_factory=dict)
    flagged_long_words: dict[int, list[str]] = field(default_factory=dict)
    product_names: list[str] = field(default_factory=list)
    user_names: list[str] = field(default_factory=list)
    email_reviews: int = 0

    def __len__(self) -> int:
        return int(self.rating.size)

    def kept_by_step_3(self) -> np.ndarray:
        """Reviews kept by the locked step 3 rules: 1+ words, at most MAX_TOKENS tokens."""
        return (self.words >= 1) & (self.tokens <= MAX_TOKENS)

    def english(self) -> np.ndarray:
        """True where fastText's top guess is English."""
        if "en" not in self.languages:
            return np.zeros(len(self), dtype=bool)
        return self.language == self.languages.index("en")

    def word_set(self, index: int) -> np.ndarray:
        return self.word_ids[self.word_offsets[index] : self.word_offsets[index + 1]]


def build_review_table(path: Path, identifier: LanguageIdentifier | None = None) -> ReviewTable:
    """Read the raw review file once and collect a ReviewTable.

    Args:
        path: Raw review file in the Amazon Reviews 2023 format.
        identifier: Language identifier; a new one is made if omitted.

    Returns:
        The table, with word evidence fitted on confident long reviews.
    """
    identifier = identifier or LanguageIdentifier()
    ids: dict[str, dict] = {"product": {}, "user": {}, "text": {}, "language": {NO_LANGUAGE: 0}}
    # Keys are product and user ids, text digests (bytes) and language codes.
    vocabulary: dict[str, int] = {}
    columns = {
        "product": array("i"),
        "user": array("i"),
        "timestamp": array("q"),
        "rating": array("b"),
        "words": array("i"),
        "tokens": array("i"),
        "text_id": array("i"),
        "language": array("h"),
        "probability": array("f"),
        "runner_up": array("h"),
        "runner_up_probability": array("f"),
    }
    word_ids, word_offsets = array("I"), array("q", [0])
    short_reviews: dict[int, ShortReview] = {}
    flagged: dict[int, str] = {}
    foreign_sentences: dict[int, list[ForeignSentence]] = {}
    flagged_long_words: dict[int, list[str]] = {}
    markup_reviews: Counter[str] = Counter()
    markup_examples: dict[str, list[str]] = {}
    email_reviews = 0
    evidence = WordEvidence()

    def intern(kind: str, key: str | bytes) -> int:
        table = ids[kind]
        found = table.get(key)
        if found is None:
            found = table[key] = len(table)
        return found

    for index, record in enumerate(iter_reviews(path)):
        raw = record.get("text") or ""
        seen_markup: set[str] = set()
        for match in _MARKUP.finditer(raw):
            name = match.group(1).upper()
            if name in seen_markup:
                continue
            seen_markup.add(name)
            markup_reviews[name] += 1
            found = markup_examples.setdefault(name, [])
            if len(found) < MARKUP_EXAMPLES:
                closing = raw.find("]]", match.start())
                end = closing + 2 if closing != -1 else match.start() + EXCERPT_CHARS
                found.append(raw[match.start() : min(end, match.start() + EXCERPT_CHARS)])
        email_reviews += has_email(raw)
        tokens = tokenize(clean_text(raw))
        word_tokens = [token for token in tokens if token[0].isalpha()]
        digest = hashlib.blake2b(" ".join(tokens).encode("utf-8"), digest_size=12).digest()

        columns["product"].append(
            intern("product", record.get("parent_asin") or record.get("asin") or "")
        )
        columns["user"].append(intern("user", record.get("user_id") or ""))
        columns["timestamp"].append(int(record.get("timestamp") or 0))
        columns["rating"].append(int(round(float(record["rating"]))))
        columns["words"].append(len(word_tokens))
        columns["tokens"].append(len(tokens))
        columns["text_id"].append(intern("text", digest))

        if word_tokens:
            prepared = text_for_language_id(raw)
            guess = identifier.guess(prepared)
            columns["language"].append(intern("language", guess.language))
            columns["probability"].append(guess.probability)
            columns["runner_up"].append(intern("language", guess.runner_up))
            columns["runner_up_probability"].append(guess.runner_up_probability)
            if not guess.is_english:
                flagged[index] = excerpt(raw)
                if len(word_tokens) > SHORT_MAX_WORDS:
                    flagged_long_words[index] = unicode_words(prepared)
            if len(word_tokens) <= SHORT_MAX_WORDS:
                short_reviews[index] = ShortReview(unicode_words(prepared), prepared, excerpt(raw))
            else:
                if guess.probability >= CONFIDENT:
                    evidence.add(unicode_words(prepared), guess.is_english)
                distinct = sorted({vocabulary.setdefault(w, len(vocabulary)) for w in word_tokens})
                word_ids.extend(distinct)
                for sentence in sentences_for_language_id(raw, SENTENCE_MIN_WORDS):
                    found_guess = identifier.guess(sentence)
                    if (
                        not found_guess.is_english
                        and found_guess.probability >= SENTENCE_MIN_PROBABILITY
                    ):
                        foreign_sentences.setdefault(index, []).append(
                            ForeignSentence(
                                sentence,
                                found_guess.language,
                                found_guess.probability,
                                unicode_words(sentence),
                            )
                        )
        else:
            columns["language"].append(0)
            columns["probability"].append(0.0)
            columns["runner_up"].append(0)
            columns["runner_up_probability"].append(0.0)
        word_offsets.append(len(word_ids))

    if not columns["rating"]:
        raise ValueError(f"No reviews found in {path}")

    arrays = {name: np.asarray(values) for name, values in columns.items()}

    def names(kind: str) -> list[str]:
        return [str(key) for key, _ in sorted(ids[kind].items(), key=lambda item: item[1])]

    languages = names("language")
    return ReviewTable(
        **{name: arrays[name].astype(np.int64) for name in _INTEGER_COLUMNS},
        probability=arrays["probability"].astype(np.float64),
        runner_up_probability=arrays["runner_up_probability"].astype(np.float64),
        languages=languages,
        word_ids=np.asarray(word_ids, dtype=np.int64),
        word_offsets=np.asarray(word_offsets, dtype=np.int64),
        short_reviews=short_reviews,
        flagged_excerpts=flagged,
        evidence=evidence,
        foreign_sentences=foreign_sentences,
        markup_reviews=markup_reviews,
        markup_examples=markup_examples,
        email_reviews=email_reviews,
        flagged_long_words=flagged_long_words,
        product_names=names("product"),
        user_names=names("user"),
    )


_INTEGER_COLUMNS = (
    "product",
    "user",
    "timestamp",
    "rating",
    "words",
    "tokens",
    "text_id",
    "language",
    "runner_up",
)


def fetch_excerpts(path: Path, indices: set[int]) -> dict[int, str]:
    """Read the file again and return excerpts of the reviews at ``indices``."""
    wanted = set(indices)
    found: dict[int, str] = {}
    if not wanted:
        return found
    last = max(wanted)
    for index, record in enumerate(iter_reviews(path)):
        if index in wanted:
            found[index] = excerpt(record.get("text") or "")
        if index >= last:
            break
    return found
