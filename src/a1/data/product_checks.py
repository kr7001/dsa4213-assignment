"""Evidence for the cap on reviews per product (step 4).

Three questions:

1. How concentrated are reviews in a few products, and what would each cap
   cost in reviews and tokens?
2. How alike are reviews of the same product? Measured by the intraclass
   correlation (ICC): roughly, the correlation between the star ratings of
   two reviews of the same product. 0 means the product tells you nothing
   about the stars; 1 means all its reviews give the same stars.
3. How much does a random product-level split vary from draw to draw, with
   and without a cap? Simulated over many random 80/10/10 splits of products.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from a1.data.review_table import ReviewTable

CAPS = (10, 20, 30, 50, 100, 200, 500)
SIMULATED_CAPS: tuple[int | None, ...] = (None, 30, 100, 200)
TEST_SHARE = 0.1  # share of products in the test split of the simulated 80/10/10 splits
N_SPLITS = 200


def intraclass_correlation(values: np.ndarray, groups: np.ndarray) -> float | None:
    """One-way ICC(1) of ``values`` grouped by ``groups``, over groups with 2+ members.

    Uses the ANOVA estimator with the usual correction for unequal group sizes
    (n0), so small groups do not inflate the estimate by chance.

    Args:
        values: One number per item.
        groups: Group id per item (non-negative integers).

    Returns:
        The ICC, or None when there are fewer than two groups of 2+ items.
    """
    sizes = np.bincount(groups)
    in_pairs = sizes[groups] >= 2
    values, groups = values[in_pairs].astype(float), groups[in_pairs]
    if values.size == 0:
        return None
    _, groups = np.unique(groups, return_inverse=True)
    n, k = values.size, int(groups.max()) + 1
    if k < 2:
        return None
    sizes = np.bincount(groups).astype(float)
    means = np.bincount(groups, weights=values) / sizes
    ms_between = (sizes * (means - values.mean()) ** 2).sum() / (k - 1)
    ms_within = ((values - means[groups]) ** 2).sum() / (n - k)
    n0 = (n - (sizes**2).sum() / n) / (k - 1)
    denominator = ms_between + (n0 - 1) * ms_within
    return float((ms_between - ms_within) / denominator) if denominator else None


def summarize_products(table: ReviewTable, kept: np.ndarray, seed: int) -> dict[str, Any]:
    """Measure product concentration, within-product similarity and split stability.

    Args:
        table: The review table.
        kept: Boolean mask of reviews kept so far.
        seed: Seed for the simulated splits.

    Returns:
        Numbers only, for the summary file.
    """
    product = table.product[kept]
    n_products_all = int(table.product.max()) + 1
    counts_all = np.bincount(product, minlength=n_products_all)
    active = counts_all > 0
    counts = counts_all[active]
    tokens = np.bincount(product, weights=table.tokens[kept], minlength=n_products_all)[active]
    rating = table.rating[kept]
    positives = np.bincount(product[rating >= 4], minlength=n_products_all)[active]
    negatives = np.bincount(product[rating <= 2], minlength=n_products_all)[active]

    total = int(counts.sum())
    ranked = np.sort(counts)[::-1]

    def top_share(fraction: float) -> float:
        return float(ranked[: max(1, math.ceil(fraction * ranked.size))].sum() / total)

    labelled = rating != 3
    return {
        "products": int(counts.size),
        "reviews": total,
        "reviews_per_product": {
            "mean": float(counts.mean()),
            "median": float(np.median(counts)),
            "p90": float(np.percentile(counts, 90)),
            "p99": float(np.percentile(counts, 99)),
            "max": int(counts.max()),
        },
        "share_products_with_one_review": float((counts == 1).mean()),
        "share_reviews_in_top_1pct_products": top_share(0.01),
        "share_reviews_in_top_10pct_products": top_share(0.10),
        "largest_product_share": float(counts.max() / total),
        "cap_table": [
            {
                "cap": cap,
                "products_above_cap": int((counts > cap).sum()),
                "share_reviews_kept": float(np.minimum(counts, cap).sum() / total),
                "share_tokens_kept": float(
                    (tokens * np.minimum(1.0, cap / counts)).sum() / tokens.sum()
                ),
            }
            for cap in CAPS
        ],
        "icc": {
            "stars": intraclass_correlation(rating, product),
            "positive_vs_negative": intraclass_correlation(
                (rating[labelled] >= 4).astype(float), product[labelled]
            ),
            "meaning": "about the correlation between the ratings of two reviews of one product",
        },
        "split_simulation": _simulate_splits(counts, positives, negatives, seed),
    }


def _simulate_splits(
    counts: np.ndarray, positives: np.ndarray, negatives: np.ndarray, seed: int
) -> dict[str, Any]:
    """Draw random product-level splits and describe the test split each time.

    A cap keeps at most ``cap`` reviews per product; the positive and negative
    counts of a capped product shrink in proportion, which is what a random
    choice of reviews gives on average.
    """
    rng = np.random.default_rng(seed)
    n_test = max(1, round(TEST_SHARE * counts.size))
    capped = {}
    for cap in SIMULATED_CAPS:
        kept = counts if cap is None else np.minimum(counts, cap)
        scale = kept / counts
        capped[cap] = (kept.astype(float), positives * scale, negatives * scale)

    draws: dict[int | None, dict[str, list[float]]] = {
        cap: {"reviews": [], "largest_share": [], "positive_share": [], "equal_products": []}
        for cap in SIMULATED_CAPS
    }
    for _ in range(N_SPLITS):
        test = rng.permutation(counts.size)[:n_test]
        for cap, (kept, pos, neg) in capped.items():
            sizes = kept[test]
            total = sizes.sum()
            record = draws[cap]
            record["reviews"].append(float(total))
            record["largest_share"].append(float(sizes.max() / total))
            record["positive_share"].append(float(pos[test].sum() / (pos[test] + neg[test]).sum()))
            # The number of equal-sized products that would be as concentrated.
            record["equal_products"].append(float(total**2 / (sizes**2).sum()))

    def spread(values: list[float]) -> dict[str, float]:
        low, mid, high = np.percentile(values, [2.5, 50, 97.5])
        return {"p2.5": float(low), "median": float(mid), "p97.5": float(high)}

    return {
        "splits": N_SPLITS,
        "test_share_of_products": TEST_SHARE,
        "by_cap": [
            {"cap": "none" if cap is None else cap}
            | {name: spread(values) for name, values in draws[cap].items()}
            for cap in SIMULATED_CAPS
        ],
    }


def format_products(summary: dict[str, Any]) -> list[str]:
    """Render the product numbers as report lines."""
    per = summary["reviews_per_product"]
    icc = summary["icc"]
    lines = [
        "== Reviews per product (cap) ==",
        f"{summary['products']:,} products, {summary['reviews']:,} reviews | per product: mean "
        f"{per['mean']:.1f}, median {per['median']:g}, 90th pct {per['p90']:g}, 99th pct "
        f"{per['p99']:g}, max {per['max']:,} ({summary['largest_product_share']:.2%} of reviews)",
        f"{summary['share_products_with_one_review']:.1%} of products have 1 review; the top 1% "
        f"of products hold {summary['share_reviews_in_top_1pct_products']:.1%} of reviews, the "
        f"top 10% hold {summary['share_reviews_in_top_10pct_products']:.1%}",
        "cap | products above it | reviews kept | tokens kept",
    ]
    for row in summary["cap_table"]:
        lines.append(
            f"{row['cap']:>4} | {row['products_above_cap']:>6,} | "
            f"{row['share_reviews_kept']:>6.1%} | {row['share_tokens_kept']:>6.1%}"
        )
    stars, label = icc["stars"], icc["positive_vs_negative"]
    lines += [
        f"Ratings of the same product are alike: ICC {stars:.2f} for stars, "
        f"{label:.2f} for positive vs negative ({icc['meaning']}).",
        f"Simulated {summary['split_simulation']['splits']} random product splits; test split "
        "(10% of products), as 2.5th pct / median / 97.5th pct:",
        "cap  | test reviews | biggest product's share | positive share | as many equal products",
    ]
    for row in summary["split_simulation"]["by_cap"]:

        def fmt(stat: dict[str, float], kind: str) -> str:
            values = (stat["p2.5"], stat["median"], stat["p97.5"])
            if kind == "count":
                return " / ".join(f"{v:,.0f}" for v in values)
            return " / ".join(f"{v:.1%}" for v in values)

        lines.append(
            f"{row['cap']!s:>4} | {fmt(row['reviews'], 'count')} | "
            f"{fmt(row['largest_share'], 'share')} | {fmt(row['positive_share'], 'share')} | "
            f"{fmt(row['equal_products'], 'count')}"
        )
    return lines
