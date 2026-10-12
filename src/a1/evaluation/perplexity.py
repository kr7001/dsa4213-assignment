"""Scoring a model under the rules locked on 12 Oct (D0.5).

- Each review is scored on its own, from <bos>.
- The targets are every token after <bos>, including <eos>; padding is never
  scored. The number of targets N is checked against the data, so every model
  is scored on exactly the same guesses (P1.9).
- J_test = (1/N) * sum of -log P(x_{t+1} | x_{1:t}) over all N targets, and
  PPL = exp(J_test) (Lecture 3, slide 28).
- The share of targets that are <unk> is reported beside it, because a smaller
  vocabulary makes perplexity look better (D0.4).

It can also return J by position in the review, for the context comparison
(D1.5): the loss at target position t averaged over every review that long.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from a1.data.sequences import EncodedReviews, collate, make_batches, padded_width
from a1.data.vocab import PAD_ID, UNK_ID


@torch.no_grad()
def evaluate(
    model: nn.Module,
    reviews: EncodedReviews,
    batch_tokens: int,
    device: torch.device,
    by_position: bool = False,
) -> dict[str, Any]:
    """J, PPL, the target count and the <unk> share on these reviews."""
    model.eval()
    total = 0.0
    targets_seen = 0
    longest = padded_width(int(reviews.lengths.max()) + 1) if len(reviews) else 0
    position_sums = np.zeros(longest)
    position_counts = np.zeros(longest, dtype=np.int64)
    for index in make_batches(reviews.lengths, batch_tokens):
        inputs, targets = collate(reviews, index, device)
        logits = model(inputs)
        losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=PAD_ID,
            reduction="none",
        ).view(targets.shape)
        total += float(losses.sum())
        targets_seen += int(reviews.lengths[index].sum()) + index.size
        if by_position:
            mask = targets != PAD_ID
            width = targets.size(1)
            position_sums[:width] += losses.sum(0).cpu().double().numpy()  # MPS has no float64
            position_counts[:width] += mask.sum(0).cpu().numpy()
    if targets_seen != reviews.n_targets:
        raise AssertionError(f"Scored {targets_seen} targets, expected {reviews.n_targets}")
    j = total / targets_seen
    result: dict[str, Any] = {
        "J": j,
        "PPL": math.exp(j),
        "targets": targets_seen,
        "reviews": len(reviews),
        "unk_share": float((reviews.ids == UNK_ID).sum()) / targets_seen,
    }
    if by_position:
        with np.errstate(invalid="ignore", divide="ignore"):
            result["J_by_position"] = (position_sums / position_counts).tolist()
        result["targets_by_position"] = position_counts.tolist()
    return result
