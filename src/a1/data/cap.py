"""The cap on reviews per product (decision D0.10).

Reviews of one product resemble each other. If the star ratings of two
reviews of the same product correlate at rho (the intraclass correlation),
then m reviews of one product carry the information of

    m / (1 + (m - 1) * rho)

independent reviews (Kish, 1965, Survey Sampling), which levels off at 1 / rho
however large m gets. The cap is the smallest m that reaches INFORMATION_SHARE
of that limit. Each product above the cap keeps a random subset of its reviews.

rho is computed from the ratings of every product, including the products
that will later fall in the test pile. That uses no model results, and
:func:`cap_on_random_halves` checks that it does not matter: it recomputes rho
and the cap on many random halves of the products.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from a1.data.product_checks import intraclass_correlation

INFORMATION_SHARE = 0.95
CHECK_REPEATS = 50  # random halves of the products in the stability check


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
    # Every review gets a fixed random number from its position in the raw file,
    # so a change upstream never reshuffles which reviews another product keeps.
    keys = np.random.default_rng(seed).random(product.size)[index]
    order = np.lexsort((keys, product[index]))
    products = product[index][order]
    starts = np.flatnonzero(np.concatenate(([True], products[1:] != products[:-1])))
    sizes = np.diff(np.append(starts, order.size))
    rank = np.arange(order.size) - np.repeat(starts, sizes)
    result = np.zeros(product.size, dtype=bool)
    result[index[order[rank < cap]]] = True
    return result


def cap_on_random_halves(
    ratings: np.ndarray, products: np.ndarray, seed: int, repeats: int = CHECK_REPEATS
) -> dict[str, Any]:
    """Recompute rho and the cap on random halves of the products.

    If the cap barely moves when only a random half of the products is used,
    it does not depend on which products end up in the test pile.

    Args:
        ratings: Star rating per review (the reviews the cap is computed on).
        products: Product id per review (non-negative integers).
        seed: Seed for choosing the halves.
        repeats: How many random halves to try.

    Returns:
        The number of halves tried and the smallest, median and largest rho and
        cap over them. ``no_cap`` counts halves whose rho was 0 or below.
    """
    ids = np.unique(products)
    rng = np.random.default_rng(seed)
    in_half = np.zeros(int(products.max()) + 1, dtype=bool)
    rhos: list[float] = []
    caps: list[int] = []
    for _ in range(repeats):
        in_half[:] = False
        in_half[ids[rng.random(ids.size) < 0.5]] = True
        mask = in_half[products]
        rho = intraclass_correlation(ratings[mask], products[mask])
        if rho is None:
            continue
        rhos.append(rho)
        if 0 < rho < 1:
            caps.append(design_effect_cap(rho))

    def spread(values: list[float] | list[int]) -> dict[str, float | None]:
        if not values:
            return {"min": None, "median": None, "max": None}
        return {
            "min": float(min(values)),
            "median": float(np.median(values)),
            "max": float(max(values)),
        }

    return {
        "repeats": repeats,
        "share_of_products": 0.5,
        "icc": spread(rhos),
        "cap": spread(caps),
        "no_cap": len(rhos) - len(caps),
    }
