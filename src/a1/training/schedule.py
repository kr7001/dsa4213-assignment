"""The learning-rate plan (D0.6, locked 12 Oct).

Linear warm-up over the first 4% of steps, because Adam's early step sizes rest
on very few gradients (Liu et al., 2020; Vaswani et al., 2017 warmed up for 4%
of training). Then one cosine curve (Loshchilov and Hutter, 2017, eq. 5) down
to 10% of the peak, reached exactly at the planned last step: overshooting the
run length hurts (Hoffmann et al., 2022, appendix D.1), and decaying 10x was
slightly better than decaying to zero (their footnote 9).
"""

from __future__ import annotations

import math

import torch

WARMUP_FRACTION = 0.04
FINAL_FRACTION = 0.1


def lr_factor(
    step: int,
    total_steps: int,
    warmup_fraction: float = WARMUP_FRACTION,
    final_fraction: float = FINAL_FRACTION,
) -> float:
    """The multiple of the peak learning rate used at a 0-based step."""
    if total_steps < 1:
        raise ValueError("total_steps must be at least 1")
    warmup = max(1, round(warmup_fraction * total_steps))
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup - 1))
    return final_fraction + (1.0 - final_fraction) * 0.5 * (1.0 + math.cos(math.pi * progress))


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_fraction: float = WARMUP_FRACTION,
    final_fraction: float = FINAL_FRACTION,
) -> torch.optim.lr_scheduler.LambdaLR:
    """A scheduler whose step k uses lr_factor(k) times each group's peak rate."""
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, total_steps, warmup_fraction, final_fraction)
    )
