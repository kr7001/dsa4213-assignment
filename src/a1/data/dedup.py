"""Removing duplicates (decision D0.11).

Among the reviews still kept, taken in time order (earliest first):

1. An exact copy of an earlier review by the same person is removed, whatever
   its length: it is the same review counted again.
2. An exact copy of an earlier review of 7+ words by a different person is
   removed: long texts are almost never written twice independently.
3. A near copy by the same person (two reviews of 7+ words sharing at least
   90% of their distinct words, measured by Jaccard overlap) is removed,
   keeping the earliest review of each group of near copies.

Exact copies of 1 to 6 words by different people are kept: common phrases
such as "love it" are written independently by many people.

"Exact copy" means the same token sequence after the step 1 cleaning.
"""

from __future__ import annotations

import random

import numpy as np

from a1.data.language import SHORT_MAX_WORDS
from a1.data.review_table import ReviewTable

LONG_MIN_WORDS = SHORT_MAX_WORDS + 1
NEAR_COPY_JACCARD = 0.9
MAX_REVIEWS_PER_USER_COMPARED = 400  # bounds the pairwise work for the busiest users

KEEP, SAME_PERSON_COPY, LONG_COPY_OTHER_PERSON, SAME_PERSON_NEAR_COPY = 0, 1, 2, 3
REASONS = {
    SAME_PERSON_COPY: "exact copy by the same person",
    LONG_COPY_OTHER_PERSON: "exact copy of 7+ words by another person",
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
        ``original``: for a removed review, the index of the earlier review it
        copies (-1 otherwise), so samples can show both.
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
        is_long = table.words[first] >= LONG_MIN_WORDS
        users = {int(table.user[first])}
        for member in members[1:]:
            user = int(table.user[member])
            if user in users:
                reasons[member], original[member] = SAME_PERSON_COPY, first
            elif is_long:
                reasons[member], original[member] = LONG_COPY_OTHER_PERSON, first
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
