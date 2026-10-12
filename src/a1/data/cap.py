"""The cap on reviews per product (decision D0.10).

Reviews of one product resemble each other. If the star ratings of two
reviews of the same product correlate at rho (the intraclass correlation),
then m reviews of one product carry the information of

    m / (1 + (m - 1) * rho)

independent reviews (Kish, 1965, Survey Sampling), which levels off at 1 / rho
however large m gets. The cap is the smallest m that reaches INFORMATION_SHARE
of that limit. Each product above the cap keeps a random subset of its reviews.
"""

from __future__ import annotations

import math

import numpy as np

INFORMATION_SHARE = 0.95


def information(m: float, rho: float) -> float:
    """How many independent reviews m reviews of one product are worth."""
    return m / (1 + (m - 1) * rho)


def design_effect_cap(rho: float, share: float = INFORMATION_SHARE) -> int:
    """Smallest number of reviews whose information reaches ``share`` of 1 / rho.

    Solving m / (1 + (m - 1) rho) = share / rho for m gives
    m = share (1 - rho) / (rho (1 - share)).

    Args:
        rho: Intraclass correlation of ratings within a product, in (0, 1).
        share: Share of a product's information limit to keep, in (0, 1).

    Returns:
        The cap, at least 1.
    """
    if not 0 < rho < 1 or not 0 < share < 1:
        raise ValueError(f"Need 0 < rho < 1 and 0 < share < 1, got {rho=}, {share=}")
    exact = share * (1 - rho) / (rho * (1 - share))
    return max(1, math.ceil(exact - 1e-9))


def cap_mask(product: np.ndarray, kept: np.ndarray, cap: int, seed: int) -> np.ndarray:
    """Keep at most ``cap`` randomly chosen reviews per product.

    Args:
        product: Product id per review.
        kept: Boolean mask of the reviews still kept before this step.
        cap: Most reviews any product may keep.
        seed: Seed for the random choice.

    Returns:
        Boolean mask of the reviews kept after the cap.
    """
    index = np.flatnonzero(kept)
    keys = np.random.default_rng(seed).random(index.size)
    order = np.lexsort((keys, product[index]))
    products = product[index][order]
    starts = np.flatnonzero(np.concatenate(([True], products[1:] != products[:-1])))
    sizes = np.diff(np.append(starts, order.size))
    rank = np.arange(order.size) - np.repeat(starts, sizes)
    result = np.zeros(product.size, dtype=bool)
    result[index[order[rank < cap]]] = True
    return result
