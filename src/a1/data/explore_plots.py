"""Charts for the corpus statistics: ratings, lengths and reviews per product."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from a1.data.explore import CorpusStatistics
from a1.viz import (
    HIGH_RATING,
    INK_SECONDARY,
    LINE_WIDTH,
    LOW_RATING,
    MID_RATING,
    SERIES,
    SURFACE,
    add_titles,
    apply_style,
    use_compact_y,
    use_plain_log_x,
)

RATING_GROUPS = (
    ("1-2 stars", (1, 2), LOW_RATING),
    ("3 stars", (3,), MID_RATING),
    ("4-5 stars", (4, 5), HIGH_RATING),
)


def plot_statistics(stats: CorpusStatistics, out_dir: Path) -> list[Path]:
    """Draw every corpus chart into ``out_dir`` and return the file paths."""
    apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        _plot_ratings(stats, out_dir / "ratings.png"),
        _plot_lengths(stats, out_dir / "lengths.png"),
        _plot_products(stats, out_dir / "products.png"),
    ]


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_ratings(stats: CorpusStatistics, path: Path) -> Path:
    counts = stats.summary["ratings"]["counts"]
    shares = stats.summary["ratings"]["shares"]
    stars = np.arange(1, 6)
    values = [counts[str(star)] for star in stars]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.78, bottom=0.14)
    ax.bar(stars, values, width=0.5, color=SERIES[0])
    for star, value in zip(stars, values, strict=True):
        ax.annotate(
            f"{shares[str(star)] * 100:.1f}%",
            (star, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=INK_SECONDARY,
        )
    ax.set_xticks(stars, labels=[f"{star} star{'s' if star > 1 else ''}" for star in stars])
    ax.set_ylabel("Reviews")
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.12)
    use_compact_y(ax)
    add_titles(
        fig,
        "Star ratings",
        f"{stats.summary['size']['reviews']:,} reviews; label is the share of all reviews",
    )
    return _save(fig, path)


def _plot_lengths(stats: CorpusStatistics, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.8, bottom=0.14)
    for label, stars, color in RATING_GROUPS:
        subset = np.sort(stats.lengths[np.isin(stats.ratings, stars)])
        if subset.size == 0:
            continue
        values = np.unique(subset[subset > 0])
        at_or_below = np.searchsorted(subset, values, side="right") / subset.size
        ax.step(
            values,
            at_or_below * 100,
            where="post",
            color=color,
            linewidth=LINE_WIDTH,
            label=f"{label} ({subset.size:,} reviews)",
        )
    ax.set_xscale("log")
    use_plain_log_x(ax)
    ax.set_xlim(left=1)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Review length in course tokens (log scale)")
    ax.set_ylabel("Reviews at or below this length (%)")
    ax.legend(loc="lower right")
    zero_share = float((stats.lengths == 0).mean())
    add_titles(
        fig,
        "Review length by star rating",
        f"Read across: the share of reviews no longer than a given length. "
        f"{zero_share * 100:.2f}% have 0 tokens (not shown).",
    )
    return _save(fig, path)


def _plot_products(stats: CorpusStatistics, path: Path) -> Path:
    counts = stats.reviews_per_product
    total = counts.sum()
    ranked = np.sort(counts)[::-1]
    held = np.cumsum(ranked) / total
    top_fraction = np.arange(1, len(ranked) + 1) / len(ranked)

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.15, wspace=0.28)

    left.plot(top_fraction * 100, held * 100, color=SERIES[0], linewidth=LINE_WIDTH)
    for fraction in (0.01, 0.10):
        k = max(1, int(np.ceil(fraction * len(ranked))))
        share = held[k - 1] * 100
        left.plot(
            fraction * 100,
            share,
            "o",
            markersize=7,
            color=SERIES[0],
            markeredgecolor=SURFACE,
            markeredgewidth=2,
        )
        left.annotate(
            f"top {fraction:.0%} hold {share:.0f}%",
            (fraction * 100, share),
            xytext=(8, -12),
            textcoords="offset points",
            fontsize=8.5,
            color=INK_SECONDARY,
        )
    left.set_xscale("log")
    use_plain_log_x(left, suffix="%")
    left.set_xlim(100 / len(ranked), 100)
    left.set_ylim(0, 100)
    left.set_xlabel("Top X% of products by review count (log scale)")
    left.set_ylabel("Share of all reviews (%)")
    left.set_title("How concentrated are the reviews?", fontsize=9.5, loc="left")

    caps = np.unique(np.round(np.logspace(0, np.log10(max(2, ranked[0])), 80)).astype(int))
    kept = np.array([np.minimum(counts, cap).sum() / total for cap in caps])
    right.plot(caps, kept * 100, color=SERIES[0], linewidth=LINE_WIDTH)
    right.set_xscale("log")
    use_plain_log_x(right)
    right.set_ylim(0, 100)
    right.set_xlabel("Cap: most reviews kept per product (log scale)")
    right.set_ylabel("Reviews kept (%)")
    right.set_title("What would a per-product cap cost?", fontsize=9.5, loc="left")

    add_titles(
        fig,
        "How reviews spread across products",
        f"{len(counts):,} products, {total:,} reviews; median "
        f"{np.median(counts):g} review(s) per product, largest {ranked[0]:,}",
    )
    return _save(fig, path)
