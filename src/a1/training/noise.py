"""The gradient noise scale, which tells us how large a batch is still useful.

McCandlish et al. (2018) show that the "simple noise scale"
B_simple = tr(Sigma) / |G|^2 "predicts the largest useful batch size": below
it, a bigger batch cuts the steps needed almost in proportion; above it, the
extra text per update is mostly wasted. Their appendix A.1 (eq. A.2) estimates
both parts, without bias, from squared gradient norms measured at a small and
a big batch size:

    |G|^2     ~ (B_big |G_big|^2 - B_small |G_small|^2) / (B_big - B_small)
    tr(Sigma) ~ (|G_small|^2 - |G_big|^2) / (1 / B_small - 1 / B_big)

Here a small batch is one micro-batch and the big batch is several micro-batches
together, all at the same parameters, so one GPU is enough. Batch sizes are
counted in targets (tokens), because the loss is a mean over targets.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from a1.data.sequences import EncodedReviews, collate, make_batches
from a1.data.vocab import PAD_ID


@dataclass
class NoiseEstimate:
    gradient_sq: float  # |G|^2, the squared size of the true gradient
    trace: float  # tr(Sigma), the noise of a single target's gradient
    b_small: float  # targets per small batch (mean)
    b_big: float  # targets in the big batch

    @property
    def b_simple(self) -> float:
        return self.trace / self.gradient_sq if self.gradient_sq > 0 else float("inf")

    def as_dict(self) -> dict[str, float]:
        return asdict(self) | {"b_simple": self.b_simple}


def simple_noise_scale(
    small_sq_norms: Sequence[float], big_sq_norm: float, b_small: float, b_big: float
) -> NoiseEstimate:
    """McCandlish et al. (2018), eq. A.2, from |G_small|^2 values and one |G_big|^2."""
    if b_big <= b_small:
        raise ValueError("The big batch must hold more targets than a small one")
    small = float(np.mean(small_sq_norms))
    gradient_sq = (b_big * big_sq_norm - b_small * small) / (b_big - b_small)
    trace = (small - big_sq_norm) / (1.0 / b_small - 1.0 / b_big)
    return NoiseEstimate(gradient_sq, trace, b_small, b_big)


def measure_noise_scale(
    model: nn.Module,
    reviews: EncodedReviews,
    micro_tokens: int,
    micro_batches: int,
    device: torch.device,
    seed: int,
) -> NoiseEstimate:
    """One noise-scale measurement at the model's current parameters (no update)."""
    batches = make_batches(reviews.lengths, micro_tokens, seed=seed)[:micro_batches]
    if len(batches) < 2:
        raise ValueError("Need at least two micro-batches")
    params = [p for p in model.parameters() if p.requires_grad]
    total = [torch.zeros_like(p) for p in params]
    small_sq: list[float] = []
    counts: list[int] = []
    model.train()
    for index in batches:
        inputs, targets = collate(reviews, index, device)
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=PAD_ID
        )
        model.zero_grad(set_to_none=True)
        loss.backward()
        n = int(reviews.lengths[index].sum()) + index.size
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
        small_sq.append(float(sum((g.float() ** 2).sum() for g in grads)))
        for acc, g in zip(total, grads, strict=True):
            acc.add_(g, alpha=n)
        counts.append(n)
    model.zero_grad(set_to_none=True)
    big_sq = float(sum(((acc / sum(counts)) ** 2).sum() for acc in total))
    # Each small gradient is a mean over its own n targets, so its expected squared norm
    # is |G|^2 + tr(Sigma) / n; averaging over micro-batches of slightly different sizes
    # makes the right B_small the harmonic mean of their sizes.
    b_small = len(counts) / sum(1.0 / n for n in counts)
    return simple_noise_scale(small_sq, big_sq, b_small, float(sum(counts)))
