"""J by target position in the review (D1.5, locked 14 Oct).

Position 1 is the first word after <bos>. For targets 1 to 4 the 5-gram sees the
whole review so far, as the neural models do, so a gap between models there
cannot come from context length; from position 5 on the neural models see
more. Positions are grouped so that every group holds enough targets: the
first group is the 5-gram's window, then each group doubles in width.
Models are compared with each other inside a group, never one model's early
groups with its late ones (late positions exist only in long reviews).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from a1.data.sequences import EncodedReviews
from a1.models.ngram import flat_sequences

POSITION_GROUPS = ((1, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 513))


def group_positions(
    sums: np.ndarray, counts: np.ndarray, groups: tuple[tuple[int, int], ...] = POSITION_GROUPS
) -> list[dict[str, Any]]:
    """J per group of positions; sums[i] and counts[i] belong to position i + 1."""
    sums = np.nan_to_num(np.asarray(sums, dtype=np.float64))
    counts = np.asarray(counts, dtype=np.int64)
    rows = []
    for low, high in groups:
        total = float(sums[low - 1 : high].sum())
        n = int(counts[low - 1 : high].sum())
        rows.append({"positions": [low, high], "targets": n, "J": total / n if n else None})
    return rows


def neural_position_sums(result: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Per-position loss sums and target counts from evaluate(..., by_position=True)."""
    counts = np.asarray(result["targets_by_position"], dtype=np.int64)
    means = np.nan_to_num(np.asarray(result["J_by_position"], dtype=np.float64))
    return means * counts, counts


def ngram_position_sums(
    log_probs: np.ndarray, reviews: EncodedReviews
) -> tuple[np.ndarray, np.ndarray]:
    """The same sums for an n-gram model's log_probs (targets in review order)."""
    _, offsets = flat_sequences(reviews)
    positions = offsets[offsets >= 1]
    if positions.size != log_probs.size:
        raise AssertionError("log_probs and reviews disagree on the number of targets")
    longest = int(positions.max()) if positions.size else 0
    sums = np.bincount(positions - 1, weights=-np.asarray(log_probs), minlength=longest)
    counts = np.bincount(positions - 1, minlength=longest)
    return sums, counts


def mean_context(reviews: EncodedReviews) -> float:
    """Average number of tokens visible to a prediction (<bos> plus the words before it).

    A review of n words has targets at positions 1 .. n + 1, and the target at
    position t sees t tokens, so the review contributes (n + 1)(n + 2) / 2.
    """
    n = reviews.lengths.astype(np.float64)
    return float(((n + 1) * (n + 2) / 2).sum() / (n + 1).sum())
