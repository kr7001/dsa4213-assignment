"""Build the report's data figures and tables from saved result files only.

Usage:
    uv run python scripts/make_report_figures.py [--out report]

It reads counts-only result files (no review text):
results/eda/{short_reviews,filters,language_rule,vocabulary}.json,
results/preprocess/{report,corpus_checks}.json, results/data/piles.json,
results/ngram/ngram.json and results/pilot/pilot.json. It writes:

- figures/preprocessing.pdf: the measurement behind each preprocessing threshold
  (Appendix A);
- figures/batch_size.pdf: training speed and memory by batch size in the test
  run (Appendix B);
- tables/*.tex: the tables whose numbers come straight from those files.

Every number in these figures and tables is read from a file, never typed, so
the report cannot drift from the results.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

from a1.viz import (
    INK,
    INK_MUTED,
    INK_SECONDARY,
    MODEL_COLOURS,
    PAGE_WIDTH,
    SERIES,
    apply_paper_style,
)

RESULTS = Path("results")
ACCENT = SERIES[0]  # measured data in single-series panels
SIZE_STYLE = {"S": {"ls": "-", "fill": True}, "M": {"ls": "--", "fill": False}}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def n(value: float) -> str:
    """An integer with thousands separators."""
    return f"{round(value):,}"


def pct(share: float, digits: int = 1) -> str:
    """A share as a LaTeX percentage."""
    return f"{100 * share:.{digits}f}\\%"


def millions(value: float) -> str:
    return f"{value / 1e6:.2f}M"


# ---------------------------------------------------------------- figures


def label_end(ax, x: float, y: float, text: str, colour: str, dy: float = 0.0) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=(4, dy),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=7,
        color=colour,
    )


def panel_minimum_length(ax, short: dict[str, Any]) -> None:
    rows = short["minimum_words"]
    k = [r["at_least_words"] for r in rows]
    positive = [100 * r["share_positives_removed"] for r in rows]
    negative = [100 * r["share_negatives_removed"] for r in rows]
    tokens = [100 * r["share_tokens_removed"] for r in rows]
    top = max(positive) * 1.12
    ax.axvline(1, color=INK, lw=0.8, zorder=1)
    ax.text(1.2, top * 0.965, "ours: k = 1", fontsize=6.5, color=INK, va="top")
    ax.plot(k, positive, color=ACCENT, marker="o", ms=3)
    ax.plot(k, negative, color=INK_SECONDARY, marker="o", ms=3)
    ax.plot(k, tokens, color=INK_MUTED, marker="o", ms=2.5, lw=1.0)
    label_end(ax, k[-1], positive[-1], "positive reviews", ACCENT)
    label_end(ax, k[-1], negative[-1], "negative reviews", INK_SECONDARY)
    label_end(ax, k[-1], tokens[-1], "all tokens", INK_MUTED)
    three = k.index(3)
    ax.annotate(
        f"k = 3 removes {positive[three]:.1f}% of\npositives, {negative[three]:.1f}% of negatives",
        (3, positive[three]),
        xytext=(1.25, top * 0.62),
        textcoords="data",
        fontsize=6.5,
        color=INK,
        va="bottom",
        arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.5, "shrinkB": 2.5},
    )
    ax.set_xticks(k)
    ax.set_xlim(0.5, 15)
    ax.set_ylim(0, top)
    ax.set_xlabel("Rule: keep reviews with at least k words")
    ax.set_ylabel("Removed by the rule (%)")
    ax.set_title("(a) A length minimum drops positives faster")


def panel_near_copies(ax, checks: dict[str, Any]) -> None:
    near = checks["duplicates"]["near_identical"]
    by_length = {int(k): v for k, v in near["by_words_of_longer_review"].items()}
    labels = [str(w) for w in range(7, 14)] + ["14\u201326", "27+"]
    counts = [by_length.get(w, 0) for w in range(7, 14)]
    counts.append(sum(v for w, v in by_length.items() if 14 <= w <= 26))
    counts.append(sum(v for w, v in by_length.items() if w >= 27))
    x = np.array([*range(7), 7.9, 9.1])  # a gap marks the two grouped bins
    ax.bar(x, counts, color=ACCENT, width=0.72)
    for xi, c in zip(x, counts, strict=True):
        ax.text(xi, c + 3, str(c), ha="center", va="bottom", fontsize=6.5, color=INK_SECONDARY)
    line = 7.15
    ax.axvline(line, color=INK, lw=0.8)
    ax.text(
        line - 0.15,
        max(counts) * 1.1,
        "ours: from 14 words, exact\ncopies by others are removed",
        fontsize=6.5,
        color=INK,
        ha="right",
        va="top",
    )
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.6, 9.7)
    ax.set_ylim(0, max(counts) * 1.18)
    ax.set_xlabel("Words in the longer review of the pair")
    ax.set_ylabel("Pairs by different people")
    ax.set_title("(b) No near-identical pairs from 14 to 26 words")
    ax.grid(axis="x", visible=False)


def panel_cap(ax, report: dict[str, Any]) -> None:
    cap = report["cap"]
    rho = cap["icc_stars"]
    limit = cap["information_limit"]
    m = np.unique(np.round(np.logspace(0, 3, 400)).astype(int))
    effective = m / (1 + (m - 1) * rho)
    halves = cap["check_random_halves"]["cap"]
    ax.axvspan(halves["min"], halves["max"], color=INK, alpha=0.18, lw=0)
    ax.axhline(limit, color=INK_MUTED, lw=0.8)
    ax.plot(m, effective, color=ACCENT)
    ax.plot([cap["cap"]], [cap["information_at_cap"]], "o", color=INK, ms=3.5, zorder=3)
    ax.annotate(
        f"ours: cap {cap['cap']} reaches {100 * cap['information_share']:.0f}% of\n"
        f"the limit ({halves['min']:.0f}\u2013{halves['max']:.0f} on 50 random\n"
        "halves of the products)",
        (cap["cap"], cap["information_at_cap"]),
        xytext=(14, limit * 0.3),
        textcoords="data",
        fontsize=6.5,
        color=INK,
        va="bottom",
        arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.5, "shrinkB": 2.5},
    )
    ax.text(1.2, limit + 0.12, f"limit 1/\u03c1 = {limit:.2f}", fontsize=6.5, color=INK_SECONDARY)
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0, limit * 1.15)
    ax.set_xlabel("Reviews kept for one product, m")
    ax.set_ylabel("Worth in independent reviews")
    ax.set_title(f"(c) Ratings correlate (\u03c1 = {rho:.2f}), so value levels off")


def panel_vocabulary(ax, vocab: dict[str, Any]) -> None:
    rows = vocab["by_min_count"]["built_on_a_tested_on_b"]
    size = [r["vocabulary_size"] for r in rows]
    unk = [100 * r["held_out_unk_share"] for r in rows]
    ax.plot(size, unk, color=ACCENT, marker="o", ms=3)
    for r, s, u in zip(rows, size, unk, strict=True):
        chosen = r["min_count"] == 5
        ax.annotate(
            f"{r['min_count']}+" + (f" (ours): |V| = {s:,}, {u:.2f}%" if chosen else ""),
            (s, u),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=6.5,
            color=INK if chosen else INK_SECONDARY,
        )
        if chosen:
            ax.plot([s], [u], "o", color=INK, ms=3.5, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks([5000, 10000, 20000, 50000], ["5K", "10K", "20K", "50K"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(4000, 80000)
    ax.set_ylim(0, max(unk) * 1.15)
    ax.set_xlabel("Vocabulary size |V| (label: minimum count)")
    ax.set_ylabel("<unk> on unseen products (%)")
    ax.set_title("(d) Rare words add size, not coverage")


def figure_preprocessing(out: Path) -> None:
    short = load(RESULTS / "eda/short_reviews.json")
    checks = load(RESULTS / "preprocess/corpus_checks.json")
    report = load(RESULTS / "preprocess/report.json")
    vocab = load(RESULTS / "eda/vocabulary.json")
    fig, axes = plt.subplots(2, 2, figsize=(PAGE_WIDTH, 4.4))
    panel_minimum_length(axes[0, 0], short)
    panel_near_copies(axes[0, 1], checks)
    panel_cap(axes[1, 0], report)
    panel_vocabulary(axes[1, 1], vocab)
    fig.tight_layout(h_pad=1.6, w_pad=2.0)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)


def figure_batch_size(out: Path) -> None:
    pilot = load(RESULTS / "pilot/pilot.json")
    fig, (speed_ax, memory_ax) = plt.subplots(
        1, 2, figsize=(PAGE_WIDTH, 2.45), gridspec_kw={"width_ratios": [1.5, 1]}
    )
    names = {"rnn": "RNN", "lstm": "LSTM", "transformer": "Transformer"}
    memory = []
    families = list(names)
    order = sorted(
        pilot["models"],
        key=lambda m: (m["name"].split("-")[1] != "S", families.index(m["config"]["family"])),
    )
    for model in order:
        family, size = model["config"]["family"], model["name"].split("-")[1]
        rows = model["speed"]
        tokens = [r["batch_tokens"] for r in rows]
        speed = [r["targets_per_second"] / 1000 for r in rows]
        style = SIZE_STYLE[size]
        colour = MODEL_COLOURS[family]
        speed_ax.plot(
            tokens,
            speed,
            ls=style["ls"],
            marker="o",
            ms=3,
            color=colour,
            markerfacecolor=colour if style["fill"] else "white",
            label=f"{names[family]}-{size}",
        )
        memory.append([r["memory_bytes"] / 2**30 for r in rows])
    for ax in (speed_ax, memory_ax):
        ax.set_xscale("log", base=2)
        ax.set_xticks(tokens, [f"{t // 1024}K" for t in tokens])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.axvline(8192, color=INK, lw=0.8)
        ax.set_xlabel("Batch size (token slots)")
    speed_ax.text(8192 * 1.07, 49, "ours: 8K", fontsize=6.5, color=INK, va="top")
    speed_ax.set_ylabel("Training speed (K targets/s)")
    speed_ax.set_ylim(0, 50)
    speed_ax.set_title("(a) Speed peaks at 8K for five of six models")
    memory = np.array(memory)
    memory_ax.fill_between(
        tokens, memory.min(axis=0), memory.max(axis=0), color=INK_MUTED, alpha=0.3, lw=0
    )
    memory_ax.plot(tokens, memory.max(axis=0), color=INK_SECONDARY, lw=1.0)
    total = int(pilot["machine"]["memory_bytes"]) / 2**30
    memory_ax.axhline(total, color=INK_SECONDARY, lw=0.8, ls=(0, (4, 2)))
    memory_ax.text(2048, total - 0.5, f"{total:.0f} GiB memory", fontsize=6.5, color=INK, va="top")
    memory_ax.text(8192 * 1.07, 1.0, "ours: 8K", fontsize=6.5, color=INK)
    memory_ax.set_ylim(0, total * 1.12)
    memory_ax.set_ylabel("Peak memory (GiB)")
    memory_ax.set_title("(b) Peak memory nears the limit at 32K")
    handles, labels = speed_ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=6,
        fontsize=6.5,
        handlelength=2.4,
        columnspacing=1.4,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(w_pad=2.0, rect=(0, 0.07, 1, 1))
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- tables


def table_corpus(out: Path) -> None:
    """Table 1: the filter cascade, then the piles. Shares are of the reviews entering a step."""
    report = load(RESULTS / "preprocess/report.json")
    piles = load(RESULTS / "data/piles.json")["piles"]
    sizes = load(RESULTS / "data/pile_sizes.json")["slices"]
    stages = [s["reviews"] for s in report["stages"]]
    removed = [before - after for before, after in zip(stages, stages[1:], strict=False)]
    # The cascade must agree with each step's own count.
    assert removed == [
        sum(report["lengths"].values()),
        report["english_only"]["removed"],
        report["duplicates"]["removed"],
        report["cap"]["removed"],
    ], removed
    assert stages[-1] == report["final"]["reviews"]
    steps = (
        "no word, or over 512 tokens",
        "not English",
        "duplicates",
        f"over {report['cap']['cap']} per product",
    )

    def balanced(low: float, high: float) -> int:
        part = [s for s in sizes if low - 1e-9 <= s["keys"][0] and s["keys"][1] <= high + 1e-9]
        return 2 * min(sum(s["negative"] for s in part), sum(s["positive"] for s in part))

    def keys(name: str) -> str:
        low, high = piles[name]["split_keys"]
        return f"$[{low:g}, {high:g})$"

    lines = [
        r"\begin{tabular}{@{}lrr@{}}",
        r"\toprule",
        r"Corpus step & Reviews & Share \\",
        r"\midrule",
        f"Raw All\\_Beauty reviews & {n(stages[0])} & \\\\",
        *[
            f"\\quad $-$ {step} & {n(r)} & {pct(r / before)} \\\\"
            for step, r, before in zip(steps, removed, stages, strict=False)
        ],
        f"Clean corpus & {n(stages[-1])} & {pct(stages[-1] / stages[0])} \\\\",
        r"\midrule",
        r"Pile (split keys) & Reviews & Size \\",
        r"\midrule",
        f"LM training {keys('lm_train')} & {n(piles['lm_train']['reviews'])} & "
        f"{millions(piles['lm_train']['targets'])} targets \\\\",
        f"LM validation {keys('lm_valid')} & {n(piles['lm_valid']['reviews'])} & "
        f"{millions(piles['lm_valid']['targets'])} targets \\\\",
        f"LM test {keys('lm_test')} & {n(piles['lm_test']['reviews'])} & "
        f"{millions(piles['lm_test']['targets'])} targets \\\\",
        f"Part III test {keys('part3_test')} & {n(piles['part3_test']['reviews'])} & "
        f"{n(balanced(*piles['part3_test']['split_keys']))} bal. \\\\",
        f"Part III training {keys('part3_train')} & {n(piles['part3_train']['reviews'])} & "
        f"{n(balanced(*piles['part3_train']['split_keys']))} bal. \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_language(out: Path) -> None:
    rule = load(RESULTS / "eda/language_rule.json")
    lines = [
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Words & Reviews & fastText & Our rule & Flags kept \\",
        r"\midrule",
    ]
    for row in rule["by_length_group"]:
        label = "All" if row["group"] == "all" else row["group"].replace(" words", "")
        label = label.replace("-", "–")
        if row["group"] == "all":
            lines.append(r"\midrule")
        lines.append(
            f"{label} & {n(row['reviews'])} & {n(row['fasttext_alone_would_remove'])} & "
            f"{n(row['removed'])} & {n(row['fasttext_flags_now_kept'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_pilot(out: Path) -> None:
    """Table B1: what the test run measured for each candidate model."""
    pilot = load(RESULTS / "pilot/pilot.json")
    names = {"rnn": "RNN", "lstm": "LSTM", "transformer": "Transformer"}
    lines = [
        r"\begin{tabular}{@{}lrrrrrrrr@{}}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{K targets/s} & Memory & Noise & Mean & Clipped & $J$ after \\",
        r"\cmidrule(lr){3-4}",
        r"Model & Core & 8K & 16K & 8K (GiB) & scale & norm & steps & 300 steps \\",
        r"\midrule",
    ]
    for model in pilot["models"]:
        speed = {r["batch_tokens"]: r for r in model["speed"]}
        norms = model["short_run"]["gradient_norms"]
        noise = model["noise_scale"][-1]["b_simple"]
        family, size = model["config"]["family"], model["name"].split("-")[1]
        lines.append(
            f"{names[family]}-{size} & {n(model['parameters']['core'])} & "
            f"{speed[8192]['targets_per_second'] / 1000:.1f} & "
            f"{speed[16384]['targets_per_second'] / 1000:.1f} & "
            f"{speed[8192]['memory_bytes'] / 2**30:.1f} & "
            f"{n(noise) if math.isfinite(noise) else '--'} & "
            f"{norms['late_mean']:.2f} & {100 * norms['late_share_clipped']:.0f}\\% & "
            f"{model['validation_pass']['J_after_short_run']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_models(out: Path) -> None:
    """Table 2: the Part I models. Neural sizes are core parameters, counted by the model code."""
    import yaml

    from a1.models.build import build_model, count_parameters

    vocab_size = load(RESULTS / "data/piles.json")["vocabulary_size"]
    ngram = load(RESULTS / "ngram/ngram.json")
    config = yaml.safe_load(Path("configs/part1.yaml").read_text(encoding="utf-8"))
    chosen = ngram["chosen"]
    assert chosen["kneser_ney"]["order"] == chosen["add_k"]["order"]
    order = chosen["kneser_ney"]["order"]
    stored = next(
        r["stored_ngrams"]
        for r in ngram["validation"]
        if r["method"] == "kneser_ney" and r["order"] == order
    )
    rows = [
        (f"Kneser-Ney {order}-gram", f"{stored / 1e6:.1f}M $n$-grams", f"last {order - 1}", "$n$"),
        (
            f"Add-$k$ {order}-gram",
            f"{stored / 1e6:.1f}M $n$-grams",
            f"last {order - 1}",
            r"$n$, $k$, $\lambda$",
        ),
    ]
    labels = {
        "rnn-S": r"RNN, $H{=}512$",
        "lstm-S": r"LSTM, $H{=}248$",
        "transformer-S": r"Transformer",
    }
    tables, cores = set(), []
    for name in ("rnn-S", "lstm-S", "transformer-S"):
        spec = {k: v for k, v in config["models"][name].items() if k != "device"}
        counts = count_parameters(build_model(spec, vocab_size))
        tables.add(counts["word_tables"])
        cores.append(counts["core"])
        rows.append((labels[name], n(counts["core"]), "all so far", "peak LR"))
    assert len(tables) == 1, "every neural model must have the same word tables"
    lines = [
        r"\begin{tabular}{@{}llll@{}}",
        r"\toprule",
        r"Model & Size & Context & Tuned \\",
        r"\midrule",
        *[" & ".join(r) + r" \\" for r in rows],
        r"\bottomrule",
        r"\end{tabular}",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out.parent / "word_tables.tex").write_text(f"{n(tables.pop())}%\n", encoding="utf-8")
    spread = 100 * (max(cores) - min(cores)) / min(cores)
    (out.parent / "core_spread.tex").write_text(f"{spread:.1f}%\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("report"))
    args = parser.parse_args()
    missing = [
        p
        for p in (
            "eda/short_reviews.json",
            "eda/language_rule.json",
            "eda/vocabulary.json",
            "preprocess/report.json",
            "preprocess/corpus_checks.json",
            "data/piles.json",
            "data/pile_sizes.json",
            "ngram/ngram.json",
            "pilot/pilot.json",
        )
        if not (RESULTS / p).exists()
    ]
    if missing:
        print(f"Missing result files: {', '.join(missing)}", file=sys.stderr)
        return 1
    apply_paper_style()
    figures, tables = args.out / "figures", args.out / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    figure_preprocessing(figures / "preprocessing.pdf")
    figure_batch_size(figures / "batch_size.pdf")
    table_corpus(tables / "corpus.tex")
    table_language(tables / "language.tex")
    table_pilot(tables / "pilot.tex")
    table_models(tables / "models.tex")
    print(f"Wrote figures to {figures} and tables to {tables}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
