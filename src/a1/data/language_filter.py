"""The English-only rule (decision D0.9), applied to a review table.

Two signals, each used where it is reliable, and made to check each other:

- fastText's language identifier, good on longer text;
- a word score learnt from our own reviews (language.WordEvidence): the sum,
  over a text's words, of how much more typical each word is of non-English
  reviews than of English ones (log scale; 2 means about 7 times).

Rules:

1. Reviews of 1 to 6 words are judged by the word score alone, because
   fastText has too little text to go on: removed when the score is 2 or more.
2. Reviews of 7+ words are removed when fastText's top guess is another
   language and either fastText is very sure (probability 0.9 or more) or the
   word score agrees (2 or more).
3. Reviews of 7+ words kept by rule 2 are checked sentence by sentence:
   removed when a sentence of 5+ words is another language for fastText
   (probability 0.9 or more) and its words average a score of 2 or more.
   These are mixed reviews, such as English followed by a translation. The
   average, not the sum, keeps a foreign brand name inside an English
   sentence from counting.
"""

from __future__ import annotations

import numpy as np

from a1.data.language import (
    LONG_MIN_EVIDENCE,
    LONG_SURE_PROBABILITY,
    SENTENCE_MIN_MEAN_EVIDENCE,
    SHORT_MIN_EVIDENCE,
)
from a1.data.review_table import ReviewTable

KEEP, SHORT_FOREIGN, LONG_FOREIGN, MIXED = 0, 1, 2, 3
REASONS = {
    SHORT_FOREIGN: "1-6 words, foreign by word score",
    LONG_FOREIGN: "7+ words, foreign by fastText and confirmed",
    MIXED: "7+ words, contains a foreign sentence",
}


def english_only_reasons(table: ReviewTable) -> np.ndarray:
    """Return, per review, KEEP or the number of the rule that removes it.

    Reviews with no words get KEEP here; step 3 removes them anyway.

    Args:
        table: A review table, with word evidence already learnt.

    Returns:
        An int8 array aligned with the table.
    """
    evidence = table.evidence
    reasons = np.zeros(len(table), dtype=np.int8)

    scores: dict[str, float] = {}
    for index, review in table.short_reviews.items():
        score = scores.get(review.text)
        if score is None:
            score = scores[review.text] = evidence.score(review.words)
        if score >= SHORT_MIN_EVIDENCE:
            reasons[index] = SHORT_FOREIGN

    for index, words in table.flagged_long_words.items():
        sure = table.probability[index] >= LONG_SURE_PROBABILITY
        if sure or evidence.score(words) >= LONG_MIN_EVIDENCE:
            reasons[index] = LONG_FOREIGN

    for index, sentences in table.foreign_sentences.items():
        if reasons[index] == KEEP and any(
            evidence.mean_score(sentence.words) >= SENTENCE_MIN_MEAN_EVIDENCE
            for sentence in sentences
        ):
            reasons[index] = MIXED
    return reasons
