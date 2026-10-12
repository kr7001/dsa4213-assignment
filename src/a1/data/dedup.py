"""Removing duplicates (decision D0.11).

Among the reviews still kept, taken in time order (earliest first):

1. An exact copy of an earlier review by the same person is removed, whatever
   its length: it is the same review counted again.
2. An exact copy of an earlier review of 14+ words by a different person is
   removed: at that length, a match is very unlikely to be chance (see below).
3. A near copy by the same person (two reviews of 7+ words sharing at least
   90% of their distinct words, measured by Jaccard overlap) is removed,
   keeping the earliest review of each group of near copies.

Exact copies of 1 to 13 words by different people are kept: people write
common sentences such as "love it" or "it broke the first time I used it"
independently.

Where 14 comes from: in the clean corpus, two different people wrote
near-identical reviews (90%+ shared words, not exact copies) at every length
from 7 to 13 words, and not once from 14 to 26 words; the few pairs above that
read as copies (scripts/check_corpus.py measures this, see corpus_checks.py).

"Exact copy" means the same token sequence after the step 1 cleaning.
"""

from __future__ import annotations

import random

import numpy as np

from a1.data.language import SHORT_MAX_WORDS
from a1.data.review_table import ReviewTable

LONG_MIN_WORDS = SHORT_MAX_WORDS + 1  # near copies by the same person: 7+ words
OTHER_PERSON_MIN_WORDS = 14  # exact copies by another person: 14+ words (see above)
NEAR_COPY_JACCARD = 0.9
MAX_REVIEWS_PER_USER_COMPARED = 400  # bounds the pairwise work for the busiest users

KEEP, SAME_PERSON_COPY, OTHER_PERSON_COPY, SAME_PERSON_NEAR_COPY = 0, 1, 2, 3
REASONS = {
    SAME_PERSON_COPY: "exact copy by the same person",
    OTHER_PERSON_COPY: f"exact copy of {OTHER_PERSON_MIN_WORDS}+ words by another person",
    SAME_PERSON_NEAR_COPY: "near copy by the same person (90%+ word overlap)",
}


def duplicate_reasons(
    table: ReviewTable, kept: np.ndarray, seed: int = 4213
) -> tuple[np.ndarray, np.ndarray]:
    """Decide which of the kept reviews are duplicates.

    Args:
        table: The review table.
        kept: Boolean mask of the reviews still kept before this step.
        seed: Seed used only if a user has more long reviews than can be compared.

    Returns:
        ``reasons``: KEEP or the rule that removes each review (int8).
        ``original``: for a review whose cleaned text repeats an earlier review,
        the index of that earlier review, whether the repeat is removed or kept
        (a short copy by another person); -1 otherwise. Samples use it to show
        both texts.
    """
    reasons = np.zeros(len(table), dtype=np.int8)
    original = np.full(len(table), -1, dtype=np.int64)
    _exact_copies(table, kept, reasons, original)
    still_kept = kept & (reasons == KEEP)
    _near_copies(table, still_kept, reasons, original, random.Random(seed))
    return reasons, original


def _time_order(table: ReviewTable, indices: np.ndarray) -> np.ndarray:
    """Sort review indices by time, then by position in the file."""
    return indices[np.lexsort((indices, table.timestamp[indices]))]


def _exact_copies(
    table: ReviewTable, kept: np.ndarray, reasons: np.ndarray, original: np.ndarray
) -> None:
    index = np.flatnonzero(kept)
    order = index[np.lexsort((index, table.timestamp[index], table.text_id[index]))]
    text = table.text_id[order]
    starts = np.flatnonzero(np.concatenate(([True], text[1:] != text[:-1])))
    ends = np.append(starts[1:], order.size)
    for start, end in zip(starts, ends, strict=True):
        if end - start < 2:
            continue
        members = order[start:end]
        first = int(members[0])
        is_long = table.words[first] >= OTHER_PERSON_MIN_WORDS
        users = {int(table.user[first])}
        for member in members[1:]:
            user = int(table.user[member])
            original[member] = first
            if user in users:
                reasons[member] = SAME_PERSON_COPY
            elif is_long:
                reasons[member] = OTHER_PERSON_COPY
            users.add(user)


def _near_copies(
    table: ReviewTable,
    kept: np.ndarray,
    reasons: np.ndarray,
    original: np.ndarray,
    rng: random.Random,
) -> None:
    long = np.flatnonzero(kept & (table.words >= LONG_MIN_WORDS))
    order = long[np.argsort(table.user[long], kind="stable")]
    user = table.user[order]
    starts = np.flatnonzero(np.concatenate(([True], user[1:] != user[:-1])))
    ends = np.append(starts[1:], order.size)

    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for start, end in zip(starts, ends, strict=True):
        if end - start < 2:
            continue
        members = [int(m) for m in order[start:end]]
        if len(members) > MAX_REVIEWS_PER_USER_COMPARED:
            members = sorted(rng.sample(members, MAX_REVIEWS_PER_USER_COMPARED))
        sets = [frozenset(table.word_set(m).tolist()) for m in members]
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                size_a, size_b = len(sets[a]), len(sets[b])
                if min(size_a, size_b) < NEAR_COPY_JACCARD * max(size_a, size_b):
                    continue  # too different in size to reach the threshold
                shared = len(sets[a] & sets[b])
                if shared / (size_a + size_b - shared) >= NEAR_COPY_JACCARD:
                    parent[find(members[a])] = find(members[b])

    groups: dict[int, list[int]] = {}
    for member in parent:
        groups.setdefault(find(member), []).append(member)
    for members in groups.values():
        ordered = _time_order(table, np.asarray(members, dtype=np.int64))
        first = int(ordered[0])
        for member in ordered[1:]:
            reasons[member], original[member] = SAME_PERSON_NEAR_COPY, first
