"""Part III data: labelled reviews from the two Part III piles (D3.1, locked 11 Oct).

Labels: 1-2 stars negative, 4-5 stars positive; 3-star reviews are left out of
Part III (they stay in language-model training). Within each pile the larger
class is sampled down at random to the size of the smaller one, so the classes
are balanced and each keeps its natural mix of star levels.

Products never cross piles (D0.4): every Part III review comes from a product
that no language model was trained or tuned on. Each review keeps its product,
so the classifier's cross-validation can split by product as well (D3.3).

Length in words follows D0.8: tokens that start with a letter, so punctuation
and <num> do not count.
"""

from __future__ import annotations

import gzip
import json
from array import array
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from a1.data.clean import count_words
from a1.data.sequences import EncodedReviews, Pile
from a1.data.vocab import Vocabulary

NEGATIVE, POSITIVE = 0, 1
NEGATIVE_STARS = (1, 2)
POSITIVE_STARS = (4, 5)
LENGTH_GROUPS: tuple[tuple[int, int | None], ...] = ((1, 2), (3, 6), (7, 20), (21, None))


def label_of(stars: int) -> int | None:
    """NEGATIVE for 1-2 stars, POSITIVE for 4-5 stars, None for 3 stars."""
    if stars in NEGATIVE_STARS:
        return NEGATIVE
    if stars in POSITIVE_STARS:
        return POSITIVE
    return None


@dataclass
class LabelledReviews:
    """Encoded reviews with their label, star rating, product and length in words.

    text_hash fingerprints each review's exact text within one run (Python's
    string hash), for the copy checks; it is not stored anywhere.
    """

    reviews: EncodedReviews
    labels: np.ndarray
    stars: np.ndarray
    products: np.ndarray
    words: np.ndarray
    text_hash: np.ndarray

    def __len__(self) -> int:
        return len(self.reviews)

    def subset(self, indices: np.ndarray) -> LabelledReviews:
        """The reviews at these positions, in this order."""
        indices = np.asarray(indices, dtype=np.int64)
        return LabelledReviews(
            reviews=self.reviews.subset(indices),
            labels=self.labels[indices],
            stars=self.stars[indices],
            products=self.products[indices],
            words=self.words[indices],
            text_hash=self.text_hash[indices],
        )


@dataclass
class _Builder:
    ids: array = field(default_factory=lambda: array("i"))
    offsets: list[int] = field(default_factory=lambda: [0])
    review_ids: list[int] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    stars: list[int] = field(default_factory=list)
    products: list[int] = field(default_factory=list)
    words: list[int] = field(default_factory=list)
    text_hash: list[int] = field(default_factory=list)
    three_stars: int = 0

    def build(self) -> LabelledReviews:
        reviews = EncodedReviews(
            ids=np.frombuffer(self.ids, dtype=np.int32).copy(),
            offsets=np.asarray(self.offsets, dtype=np.int64),
            review_ids=np.asarray(self.review_ids, dtype=np.int64),
        )
        return LabelledReviews(
            reviews=reviews,
            labels=np.asarray(self.labels, dtype=np.int8),
            stars=np.asarray(self.stars, dtype=np.int8),
            products=np.asarray(self.products, dtype=np.int64),
            words=np.asarray(self.words, dtype=np.int32),
            text_hash=np.asarray(self.text_hash, dtype=np.int64),
        )


@dataclass
class Part3Reading:
    """What one pass over the corpus gives Part III.

    Attributes:
        sets: Labelled (not yet balanced) reviews of each Part III pile.
        three_stars: 3-star reviews left out, per Part III pile.
        lm_train_texts: Exact-text fingerprints of every lm_train review.
        shared_products: Products found both in a Part III pile and in another
            pile (expected 0: piles are ranges of product split keys).
    """

    sets: dict[str, LabelledReviews]
    three_stars: dict[str, int]
    lm_train_texts: set[int]
    shared_products: int


def read_part3(
    path: Path,
    part3_piles: dict[str, Pile],
    other_piles: dict[str, Pile],
    vocab: Vocabulary,
) -> Part3Reading:
    """Read the corpus once: Part III reviews with labels, plus lm_train text fingerprints."""
    product_numbers: dict[str, int] = {}
    products_by_pile: dict[str, set[int]] = {name: set() for name in [*part3_piles, *other_piles]}
    builders = {name: _Builder() for name in part3_piles}
    lm_train_texts: set[int] = set()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = row["split_key"]
            number = product_numbers.setdefault(row["product"], len(product_numbers))
            for name, pile in other_piles.items():
                if pile.contains(key):
                    products_by_pile[name].add(number)
                    if name == "lm_train":
                        lm_train_texts.add(hash(row["text"]))
                    break
            for name, pile in part3_piles.items():
                if not pile.contains(key):
                    continue
                products_by_pile[name].add(number)
                builder = builders[name]
                label = label_of(int(row["stars"]))
                if label is None:
                    builder.three_stars += 1
                    break
                tokens = row["text"].split(" ")
                builder.ids.extend(vocab.encode(tokens))
                builder.offsets.append(len(builder.ids))
                builder.review_ids.append(int(row["id"]))
                builder.labels.append(label)
                builder.stars.append(int(row["stars"]))
                builder.products.append(number)
                builder.words.append(count_words([token for token in tokens if token]))
                builder.text_hash.append(hash(row["text"]))
                break
    shared = 0
    for name in part3_piles:
        others = set().union(*(products_by_pile[o] for o in products_by_pile if o != name))
        shared += len(products_by_pile[name] & others)
    return Part3Reading(
        sets={name: builder.build() for name, builder in builders.items()},
        three_stars={name: builder.three_stars for name, builder in builders.items()},
        lm_train_texts=lm_train_texts,
        shared_products=shared,
    )


def balance(data: LabelledReviews, seed: int) -> LabelledReviews:
    """Sample the larger class down to the size of the smaller one at random (D3.1).

    Reviews stay in corpus order, and the sample within a class is uniform, so
    each class keeps its natural mix of star levels.
    """
    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(data.labels == label) for label in (NEGATIVE, POSITIVE)]
    size = min(members.size for members in classes)
    kept = [
        members if members.size == size else rng.choice(members, size=size, replace=False)
        for members in classes
    ]
    return data.subset(np.sort(np.concatenate(kept)))


def describe_set(data: LabelledReviews) -> dict[str, Any]:
    """Counts for the report: reviews, classes, products, star mix and lengths."""
    stars = Counter(int(s) for s in data.stars)
    words = data.words
    return {
        "reviews": len(data),
        "negative": int((data.labels == NEGATIVE).sum()),
        "positive": int((data.labels == POSITIVE).sum()),
        "products": int(np.unique(data.products).size),
        "stars": {str(level): stars.get(level, 0) for level in (1, 2, 4, 5)},
        "tokens": int(data.reviews.lengths.sum()),
        "words_median": float(np.median(words)) if words.size else None,
        "by_length_group": {
            group_name(group): int(group_mask(words, group).sum()) for group in LENGTH_GROUPS
        },
    }


def group_name(group: tuple[int, int | None]) -> str:
    low, high = group
    return f"{low}+" if high is None else f"{low}-{high}"


def group_mask(words: np.ndarray, group: tuple[int, int | None]) -> np.ndarray:
    """Reviews whose length in words falls in the group (inclusive bounds)."""
    low, high = group
    mask = words >= low
    return mask if high is None else mask & (words <= high)


def copy_checks(
    test: LabelledReviews, train: LabelledReviews, lm_train_texts: set[int]
) -> dict[str, Any]:
    """How many test reviews repeat a text seen before, and what a copy lookup would score.

    Products never cross piles, but short texts such as "love it" recur across
    products (D0.11 keeps copies written by different people). A language model
    reads lm_train text without labels, so a repeat there is not label leakage;
    a repeat in part3_train comes with a label, so the share of test reviews a
    classifier could answer by memory is measured here: for every test review
    whose exact text is in part3_train, the majority label of those copies
    (ties count half) is compared with the true label.
    """
    in_lm = np.fromiter((h in lm_train_texts for h in test.text_hash.tolist()), dtype=bool)
    test_texts = set(test.text_hash.tolist())
    train_votes: dict[int, list[int]] = {}
    for h, label in zip(train.text_hash.tolist(), train.labels.tolist(), strict=True):
        if h in test_texts:
            train_votes.setdefault(h, [0, 0])[label] += 1
    covered = np.zeros(len(test), dtype=bool)
    lookup_correct = 0.0
    for i, (h, label) in enumerate(zip(test.text_hash.tolist(), test.labels.tolist(), strict=True)):
        counts = train_votes.get(h)
        if counts is None:
            continue
        covered[i] = True
        if counts[0] == counts[1]:
            lookup_correct += 0.5
        else:
            lookup_correct += float(int(counts[1] > counts[0]) == label)
    by_group = {}
    for group in LENGTH_GROUPS:
        mask = group_mask(test.words, group)
        by_group[group_name(group)] = {
            "reviews": int(mask.sum()),
            "share_in_lm_train": _share(in_lm[mask].sum(), mask.sum()),
            "share_in_part3_train": _share(covered[mask].sum(), mask.sum()),
        }
    return {
        "test_reviews": len(test),
        "share_in_lm_train": _share(in_lm.sum(), len(test)),
        "share_in_part3_train": _share(covered.sum(), len(test)),
        "copy_lookup_accuracy": _share(lookup_correct, covered.sum()),
        "by_length_group": by_group,
    }


def _share(part: float, whole: float) -> float | None:
    return float(part) / float(whole) if whole else None
