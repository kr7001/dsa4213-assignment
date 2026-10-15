"""Error analysis for Part III (D3.4, locked 15 Oct).

Everything here was fixed before any mistake was read.

1. Measured groups over every test review, defined from the review's tokens
   (the shared vocabulary, so a word outside it is <unk>):
   - star level (1, 2, 4 or 5 stars);
   - negation: a negation word, or a token ending in "n't" (Socher et al.
     2013, Section 5.4, test negation as a classic hard case);
   - contrast: a contrast word such as "but" (Section 5.3, "X but Y");
   - rare words: at least one <unk>, and a heavy share of them;
   - length group in words (D0.8);
   - an exact repeat of a part3_train review (which comes with its label).
2. Seeded random samples for reading: mistakes of the main representation,
   and reviews where it and a reference disagree. Each mistake gets one main
   category from CATEGORIES; if several fit, the first in the list wins.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from a1.data.vocab import UNK_ID, Vocabulary
from a1.part3.data import NEGATIVE, LabelledReviews, group_mask, group_name
from a1.part3.stats import wilson

NEGATION = frozenset(
    {
        "not",
        "no",
        "never",
        "nothing",
        "nobody",
        "none",
        "nor",
        "neither",
        "nowhere",
        "cannot",
        # contractions written without the apostrophe, common in reviews
        "dont",
        "doesnt",
        "didnt",
        "isnt",
        "wasnt",
        "arent",
        "werent",
        "cant",
        "wont",
        "couldnt",
        "wouldnt",
        "shouldnt",
        "havent",
        "hasnt",
        "hadnt",
    }
)
CONTRAST = frozenset(
    {"but", "however", "although", "though", "except", "whereas", "nevertheless", "nonetheless"}
)
CATEGORIES = (
    "rating does not match the text",
    "mixed feelings",
    "negation missed",
    "sarcasm",
    "complaint not about the product",
    "too short or vague to tell",
    "other",
)


def star_group(stars: int) -> str:
    return f"{stars} star" if stars == 1 else f"{stars} stars"


def has_negation(tokens: list[str]) -> bool:
    return any(token in NEGATION or token.endswith("n't") for token in tokens)


def has_contrast(tokens: list[str]) -> bool:
    return any(token in CONTRAST for token in tokens)


def groups(
    test: LabelledReviews,
    vocab: Vocabulary,
    train_texts: set[int],
    length_groups: list[tuple[int, int | None]],
    heavy_unk_share: float,
) -> dict[str, np.ndarray]:
    """One boolean mask per measured group, over the test reviews in order."""
    negation = np.zeros(len(test), dtype=bool)
    contrast = np.zeros(len(test), dtype=bool)
    unk_share = np.zeros(len(test))
    for i in range(len(test)):
        ids = test.reviews.review(i)
        tokens = vocab.decode(ids.tolist())
        negation[i] = has_negation(tokens)
        contrast[i] = has_contrast(tokens)
        unk_share[i] = float((ids == UNK_ID).mean()) if ids.size else 0.0
    masks: dict[str, np.ndarray] = {"all": np.ones(len(test), dtype=bool)}
    for stars in (1, 2, 4, 5):
        masks[star_group(stars)] = test.stars == stars
    masks["negation"] = negation
    masks["no negation"] = ~negation
    masks["contrast"] = contrast
    masks["no contrast"] = ~contrast
    masks["any <unk>"] = unk_share > 0
    masks[f"<unk> share at least {heavy_unk_share:.0%}"] = unk_share >= heavy_unk_share
    for group in length_groups:
        masks[f"{group_name(group)} words"] = group_mask(test.words, group)
    repeat = np.fromiter((h in train_texts for h in test.text_hash.tolist()), dtype=bool)
    masks["repeats a part3_train review"] = repeat
    masks["new text"] = ~repeat
    return masks


def group_accuracy(correct: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, Any]:
    """Accuracy with a Wilson interval inside every group."""
    rows = {}
    for name, mask in masks.items():
        count = int(mask.sum())
        right = float(correct[mask].sum())
        rows[name] = {
            "reviews": count,
            "accuracy": right / count if count else None,
            "interval": wilson(right, count),
        }
    return rows


def disagreement(first: np.ndarray, second: np.ndarray) -> dict[str, int]:
    """How often two classifiers are right or wrong together on the same reviews."""
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    return {
        "both_right": int((first & second).sum()),
        "first_only": int((first & ~second).sum()),
        "second_only": int((~first & second).sum()),
        "both_wrong": int((~first & ~second).sum()),
    }


def draw(candidates: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """A random sample (without replacement) of review positions, kept in their order."""
    candidates = np.asarray(candidates, dtype=np.int64)
    if candidates.size <= count:
        return candidates
    return np.sort(rng.choice(candidates, size=count, replace=False))


def describe_review(
    test: LabelledReviews, i: int, masks: dict[str, np.ndarray], probabilities: dict[str, float]
) -> dict[str, Any]:
    """The numbers shown next to a sampled review (no text)."""
    flags = [
        name
        for name in ("negation", "contrast", "any <unk>", "repeats a part3_train review")
        if masks[name][i]
    ]
    return {
        "review_id": int(test.reviews.review_ids[i]),
        "stars": int(test.stars[i]),
        "label": "negative" if test.labels[i] == NEGATIVE else "positive",
        "words": int(test.words[i]),
        "flags": flags,
        "probability_positive": {name: round(float(p), 3) for name, p in probabilities.items()},
    }
