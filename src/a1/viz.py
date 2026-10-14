"""Shared chart style, so every figure in the project looks the same.

Colours come from a validated colour-blind-safe palette. Categorical series
(later: one fixed colour per model) take ``SERIES`` in order. Star ratings use
a diverging pair with a grey midpoint: red for low ratings, grey for three
stars, blue for high ratings.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # Write files only; no window is needed.

import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend choice)
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
LOW_RATING = "#e34948"
MID_RATING = "#898781"
HIGH_RATING = "#2a78d6"
LINE_WIDTH = 2.0


def apply_style() -> None:
    """Set matplotlib defaults: light surface, hairline grid, quiet axes."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK_SECONDARY,
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "legend.labelcolor": INK_SECONDARY,
            "font.size": 9.5,
        }
    )


PAPER = "#ffffff"
# One fixed colour per model in every report figure (validated: all-pairs CVD
# separation 9.2+ for the first three slots; aqua needs direct labels or a legend).
MODEL_COLOURS = {
    "transformer": SERIES[0],
    "lstm": SERIES[1],
    "rnn": SERIES[2],
    "ngram": INK_SECONDARY,
}
COLUMN_WIDTH = 3.03  # inches, ACL two-column layout
PAGE_WIDTH = 6.3


def apply_paper_style() -> None:
    """Defaults for figures printed in the report: white page, small type, hairlines.

    Fonts are embedded as TrueType (pdf.fonttype 42), as conference PDFs require.
    """
    apply_style()
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica",
                "Arial",
                "Liberation Sans",
                "TeX Gyre Heros",
                "DejaVu Sans",
            ],
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlecolor": INK,
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.fontsize": 7,
            "lines.linewidth": 1.4,
            "lines.markersize": 4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def add_titles(fig: Figure, title: str, subtitle: str) -> None:
    """Write a left-aligned title and a quieter subtitle above the plot area."""
    fig.text(0.01, 0.985, title, ha="left", va="top", fontsize=12, color=INK, weight="bold")
    fig.text(0.01, 0.92, subtitle, ha="left", va="top", fontsize=9, color=INK_SECONDARY)


def compact_number(value: float, _position: int | None = None) -> str:
    """Format an axis value compactly: 950, 12K, 1.2M."""
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M".replace(".0M", "M")
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}K"
    return f"{value:.0f}"


def use_compact_y(ax: Axes) -> None:
    """Show y-axis ticks as 0 / 50K / 100K instead of full numbers."""
    ax.yaxis.set_major_formatter(FuncFormatter(compact_number))


def use_plain_log_x(ax: Axes, suffix: str = "") -> None:
    """Label a log-scale x-axis 1, 10, 100, 1K (or 0.1%, 1%, 10%) instead of powers of ten."""

    def label(value: float, _position: int | None = None) -> str:
        text = f"{value:g}" if value < 1 else compact_number(value)
        return f"{text}{suffix}"

    ax.xaxis.set_major_formatter(FuncFormatter(label))
