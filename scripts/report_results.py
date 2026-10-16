"""Report figures and tables for the results of Parts I to III (called by make_report_figures.py).

Every number comes from a result file written by a run on the author's machine:
results/runs/*/run.json and *-tuning.json (training), results/ngram/ngram.json,
results/eval/{test,costs,generation}.json (final evaluation),
results/part2/embeddings.json, results/part3/{part3,errors}.json and
results/part3/error_categories.csv (the author's categories, no review text).
The generated-text samples table also reads results/eval/examples_generation.txt,
which is git-ignored; without it the existing table is kept.

It also writes report/facts.json: the derived numbers the report text quotes,
so scripts/check_report_numbers.py can check the text against the results.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from a1.generation import detokenize
from a1.part3.stats import wilson
from a1.viz import (
    INK,
    INK_MUTED,
    INK_SECONDARY,
    MODEL_COLOURS,
    PAGE_WIDTH,
    SERIES,
)

RESULTS = Path("results")
FAMILY = {"rnn": "RNN", "lstm": "LSTM", "transformer": "Transformer"}
PART1_S = ("rnn-S-full", "lstm-S-full", "transformer-S-full")
PART1_M = ("rnn-M-full", "lstm-M-full", "transformer-M-full")
PART1_E10 = ("rnn-S-e10-full", "lstm-S-e10-full", "transformer-S-e10-full")
TRAINING_SAMPLE_STYLE = (0, (3.2, 1.6))

# Part II settings: label, colour (validated: orange, violet, green, yellow pass the
# adjacent CVD checks; the frozen random table is a no-information reference, in grey).
PART2 = {
    "random_trainable": {
        "label": "random, trainable",
        "short": "Random, trainable",
        "colour": SERIES[1],
        "runs": ("lstm-S-full", "lstm-random-s4214-full", "lstm-random-s4215-full"),
    },
    "own_frozen": {
        "label": "own skip-gram, frozen",
        "short": "Own, frozen",
        "colour": SERIES[6],
        "runs": (
            "lstm-own-frozen-s4213-full",
            "lstm-own-frozen-s4214-full",
            "lstm-own-frozen-s4215-full",
        ),
    },
    "glove_frozen": {
        "label": "GloVe, frozen",
        "short": "GloVe, frozen",
        "colour": SERIES[5],
        "runs": (
            "lstm-glove-frozen-s4213-full",
            "lstm-glove-frozen-s4214-full",
            "lstm-glove-frozen-s4215-full",
        ),
    },
    "glove_trainable": {
        "label": "GloVe, trainable",
        "short": "GloVe, trainable",
        "colour": SERIES[3],
        "runs": ("lstm-glove-trainable-s4213-full",),
    },
    "random_frozen": {
        "label": "random, frozen",
        "short": "Random, frozen",
        "colour": INK_MUTED,
        "runs": ("lstm-random-frozen-s4213-full",),
    },
}
SHORT = {
    "rnn-S-full": "RNN-S",
    "lstm-S-full": "LSTM-S",
    "transformer-S-full": "Transformer-S",
    "rnn-M-full": "RNN-M",
    "lstm-M-full": "LSTM-M",
    "transformer-M-full": "Transformer-M",
    "rnn-S-e10-full": "RNN-S, 10 ep.",
    "lstm-S-e10-full": "LSTM-S, 10 ep.",
    "transformer-S-e10-full": "Transformer-S, 10 ep.",
    "lstm-own-frozen-s4213-full": "Own skip-gram, frozen",
    "lstm-glove-frozen-s4213-full": "GloVe, frozen",
    "lstm-glove-trainable-s4213-full": "GloVe, trainable",
    "lstm-random-frozen-s4213-full": "Random, frozen",
    "kneser-ney": "Kneser-Ney 5-gram",
    "add-k": "Add-$k$ 5-gram",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def n(value: float) -> str:
    return f"{round(value):,}"


def family_of(run: str) -> str:
    return run.split("-")[0]


def run_record(run: str) -> dict[str, Any]:
    return load(RESULTS / "runs" / run / "run.json")


def run_checks(run: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Targets seen (millions), validation J and training-sample J at every check."""
    rows = [e for e in run_record(run)["summary"]["history"] if "valid_J" in e]
    x = np.array([e["targets_seen"] for e in rows]) / 1e6
    return x, np.array([e["valid_J"] for e in rows]), np.array([e["train_sample_J"] for e in rows])


def epoch_end_valid(run: str) -> list[float]:
    """Validation J at the last check of each epoch (4 checks per epoch)."""
    record = run_record(run)
    summary = record["summary"]
    per_epoch = summary["steps_per_epoch"]
    rows = [e for e in summary["history"] if "valid_J" in e]
    out = []
    epochs = int(round(summary["epochs_done"]))
    for k in range(1, epochs + 1):
        # the check at (or just before) the end of epoch k
        candidates = [e for e in rows if e["step"] <= k * per_epoch]
        if candidates:
            out.append(candidates[-1]["valid_J"])
    return out


def tex_escape(text: str) -> str:
    replace = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = "".join(replace.get(ch, ch) for ch in text)
    out = re.sub(r"<(unk|num|bos|eos|pad)>", r"\\tok{\1}", out)
    return out


def axis_epochs(ax, epoch_millions: float, epochs: int) -> None:
    ticks = [epoch_millions * k for k in range(0, epochs + 1)]
    ax.set_xticks(ticks, [f"{t:.0f}" for t in ticks])
    ax.set_xlim(0, epoch_millions * epochs * 1.02)


# ---------------------------------------------------------------- figures


def panel_part1_curves(ax, runs: tuple[str, ...], kn_valid: float | None, epoch: float) -> None:
    epochs = 0
    for run in runs:
        family = family_of(run)
        colour = MODEL_COLOURS[family]
        x, valid, train = run_checks(run)
        ax.plot(x, valid, color=colour, lw=1.3)
        ax.plot(x, train, color=colour, lw=0.9, ls=TRAINING_SAMPLE_STYLE)
        epochs = max(epochs, math.ceil(x[-1] / epoch - 1e-6))
    if kn_valid is not None:
        ax.axhline(kn_valid, color=INK_MUTED, lw=0.9)
    axis_epochs(ax, epoch, epochs)
    ax.set_xlabel("Training targets processed (millions)")
    ax.set_ylabel("$J$ (nats per target)")


def figure_curves(out: Path, epoch: float) -> None:
    """Body figure: learning curves of Parts I and II (D1.8) and the temperature test (D1.6)."""
    ngram = load(RESULTS / "ngram/ngram.json")
    kn_valid = next(
        r["valid"]["J"]
        for r in ngram["validation"]
        if r["method"] == "kneser_ney" and r["order"] == ngram["chosen"]["kneser_ney"]["order"]
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(PAGE_WIDTH, 1.92), gridspec_kw={"width_ratios": [1, 1, 0.95]}
    )
    ax = axes[0]
    panel_part1_curves(ax, PART1_S, kn_valid, epoch)
    ax.set_ylim(3.4, 5.25)
    ax.set_title("(a) Part I: no overfitting in 6 epochs")
    handles = [
        Line2D([], [], color=MODEL_COLOURS[f], lw=1.3, label=f"{FAMILY[f]}-S")
        for f in ("rnn", "lstm", "transformer")
    ]
    handles += [
        Line2D([], [], color=INK_MUTED, lw=0.9, label="KN 5-gram"),
        Line2D([], [], color=INK_SECONDARY, lw=1.1, label="validation"),
        Line2D(
            [], [], color=INK_SECONDARY, lw=0.9, ls=TRAINING_SAMPLE_STYLE, label="training sample"
        ),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=5.6, handlelength=1.8, borderaxespad=0.2)

    ax = axes[1]
    for setting in PART2.values():
        curves = [run_checks(run) for run in setting["runs"]]
        length = min(len(c[0]) for c in curves)
        x = np.mean([c[0][:length] for c in curves], axis=0)
        valid = np.array([c[1][:length] for c in curves])
        train = np.array([c[2][:length] for c in curves])
        colour = setting["colour"]
        if len(curves) > 1:
            ax.fill_between(x, valid.min(axis=0), valid.max(axis=0), color=colour, alpha=0.22, lw=0)
        ax.plot(x, valid.mean(axis=0), color=colour, lw=1.3, label=setting["label"])
        ax.plot(x, train.mean(axis=0), color=colour, lw=0.8, ls=TRAINING_SAMPLE_STYLE)
    axis_epochs(ax, epoch, 6)
    ax.set_ylim(3.4, 5.25)
    ax.set_xlabel("Training targets processed (millions)")
    ax.set_title("(b) Part II: frozen tables end higher")
    ax.legend(loc="upper right", fontsize=5.6, handlelength=1.5, borderaxespad=0.2)

    ax = axes[2]
    generation = load(RESULTS / "eval/generation.json")
    test = generation["temperature_test"]
    chosen = generation["temperature"]
    for run in ("kneser-ney", *PART1_S):
        measured = test["measured"][run]
        temps = [float(t) for t in measured]
        gaps = [measured[t]["gap"] for t in measured]
        colour = MODEL_COLOURS["ngram" if run == "kneser-ney" else family_of(run)]
        ax.plot(temps, gaps, color=colour, lw=1.1, marker="o", ms=1.8)
    ax.axhline(0, color=INK, lw=0.7)
    ax.axvline(chosen, color=INK, lw=0.8)
    ax.axvline(0.8, color=INK_MUTED, lw=0.8, ls=(0, (3, 2)))
    for x, text, colour in (
        (chosen, f"T = {chosen:.2f}", INK),
        (0.8, "T = 0.8", INK_SECONDARY),
    ):
        ax.text(
            x - 0.012,
            2.1,
            text,
            fontsize=5.8,
            color=colour,
            rotation=90,
            ha="right",
            va="top",
        )
    ax.set_xlim(0.48, 1.22)
    ax.set_ylim(-2.5, 2.2)
    ax.set_xlabel("Sampling temperature $T$")
    ax.set_ylabel("$J_{\\mathrm{own\\ samples}} - J_{\\mathrm{real\\ text}}$ (nats)")
    ax.set_title("(c) Measured temperature")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)


def figure_curves_appendix(out: Path, epoch: float) -> None:
    """Appendix figure: learning curves of the M models and the 10-epoch runs."""
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 2.3))
    panel_part1_curves(axes[0], PART1_M, None, epoch)
    axes[0].set_ylim(3.3, 5.7)
    axes[0].set_title("(a) Size check (M), 6 epochs")
    panel_part1_curves(axes[1], PART1_E10, None, epoch)
    axes[1].set_ylim(3.3, 5.7)
    axes[1].set_title("(b) Training-length check (S), 10 epochs")
    handles = [
        Line2D([], [], color=MODEL_COLOURS[f], lw=1.3, label=FAMILY[f])
        for f in ("rnn", "lstm", "transformer")
    ]
    handles += [
        Line2D([], [], color=INK_SECONDARY, lw=1.1, label="validation"),
        Line2D(
            [], [], color=INK_SECONDARY, lw=0.9, ls=TRAINING_SAMPLE_STYLE, label="training sample"
        ),
    ]
    for ax in axes:
        ax.legend(handles=handles, loc="upper right", fontsize=6.3, handlelength=2.2)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)


def figure_positions(out: Path) -> None:
    """Appendix figure: each model's lead over Kneser-Ney by target position (D1.5)."""
    test = load(RESULTS / "eval/test.json")["models"]
    kn = test["kneser-ney"]["by_position"]
    groups = [f"{g['positions'][0]}-{g['positions'][1]}" for g in kn]
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 2.2), sharey=True)
    for ax, runs, title in (
        (axes[0], PART1_S, "(a) Size S (and 10 epochs, thin)"),
        (axes[1], PART1_M, "(b) Size M"),
    ):
        for run in runs:
            rows = test[run]["by_position"]
            lead = [k["J"] - r["J"] for k, r in zip(kn, rows, strict=True)]
            ax.plot(
                range(len(groups)),
                lead,
                color=MODEL_COLOURS[family_of(run)],
                lw=1.3,
                marker="o",
                ms=2.5,
            )
        if runs == PART1_S:
            for run in PART1_E10:
                rows = test[run]["by_position"]
                lead = [k["J"] - r["J"] for k, r in zip(kn, rows, strict=True)]
                ax.plot(range(len(groups)), lead, color=MODEL_COLOURS[family_of(run)], lw=0.7)
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_xticks(range(len(groups)), [g.replace("-", "\u2013") for g in groups])
        ax.set_xlabel("Target position in the review")
        ax.set_title(title)
    axes[0].set_ylabel("Lead over Kneser-Ney (nats)")
    handles = [
        Line2D([], [], color=MODEL_COLOURS[f], lw=1.3, marker="o", ms=2.5, label=FAMILY[f])
        for f in ("rnn", "lstm", "transformer")
    ]
    axes[0].legend(handles=handles, loc="lower right", fontsize=6.5)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)


# Figure 2: one hue per design, darker for size M and lighter for 10 epochs (with
# their own marker shapes as a second cue); the Part II tables keep their Figure 1b
# colours and share a diamond, since all of them are LSTM-S variants.
RANKED_SHADES = {
    "rnn": {"S": MODEL_COLOURS["rnn"], "M": "#0b6b4b", "e10": "#93dcc1"},
    "lstm": {"S": MODEL_COLOURS["lstm"], "M": "#9a3c14", "e10": "#f7b89b"},
    "transformer": {"S": MODEL_COLOURS["transformer"], "M": "#15427f", "e10": "#9dc2f0"},
}
RANKED_MARKERS = {"S": "o", "M": "s", "e10": "^"}
RANKED_SUFFIX = {"S": "-S", "M": "-M", "e10": "-S, 10 ep."}
RANKED_TABLES = {
    "lstm-own-frozen-s4213-full": "own_frozen",
    "lstm-glove-frozen-s4213-full": "glove_frozen",
    "lstm-glove-trainable-s4213-full": "glove_trainable",
    "lstm-random-frozen-s4213-full": "random_frozen",
}


def ranked_style(run: str) -> dict[str, Any]:
    """Marker, colours and label of one of the 13 ranked models in Figure 2."""
    if run in RANKED_TABLES:
        setting = PART2[RANKED_TABLES[run]]
        return {
            "label": setting["label"],
            "marker": "D",
            "mfc": setting["colour"],
            "mec": "white",
            "ms": 5.0,
        }
    family = family_of(run)
    variant = "e10" if "-e10-" in run else ("M" if "-M-" in run else "S")
    return {
        "label": FAMILY[family] + RANKED_SUFFIX[variant],
        "marker": RANKED_MARKERS[variant],
        "mfc": RANKED_SHADES[family][variant],
        "mec": RANKED_SHADES[family]["S"] if variant == "e10" else "white",
        "ms": 5.0 if variant == "M" else 5.6,
    }


def figure_part3(out: Path) -> None:
    """Body figure: test accuracy against test J for the 13 ranked models (D3.3).

    One panel per readout (state after the last word; average over all words), one
    marker per model, and one legend under both panels.
    """
    part3 = load(RESULTS / "part3/part3.json")
    reps = part3["representations"]
    ranking = part3["ranking"]
    models = [m["model"] for m in ranking["last"]["models"]]
    counts = reps["word_counts"]["test"]["accuracy"] * 100
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 1.92), sharey=True)
    for ax, pooling, title in (
        (axes[0], "last", "(a) State after the last word"),
        (axes[1], "mean", "(b) States averaged over all words"),
    ):
        for run in models:
            j = part3["costs"][run]["test_J"]
            accuracy = reps[f"{run}/pre_output/{pooling}"]["test"]["accuracy"] * 100
            style = ranked_style(run)
            ax.plot(
                [j],
                [accuracy],
                marker=style["marker"],
                ms=style["ms"],
                mfc=style["mfc"],
                mec=style["mec"],
                mew=0.7,
                ls="",
                zorder=3,
            )
        ax.axhline(counts, color=INK_MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.text(
            4.06,
            counts + 0.15,
            f"word counts ({counts:.1f}%)",
            fontsize=6.3,
            color=INK_SECONDARY,
            ha="right",
            va="bottom",
        )
        overall = ranking[pooling]["overall"]
        ax.text(
            3.545,
            84.0,
            f"Kendall's \u03c4 = {overall['tau']:.2f}\n"
            f"({overall['same_order']} of {overall['pairs']} pairs in the same order)",
            fontsize=6.5,
            color=INK,
            va="bottom",
            linespacing=1.3,
        )
        ax.set_title(title)
        ax.set_xlabel("Language-model test $J$")
        ax.set_xlim(3.53, 4.07)
    axes[0].set_ylabel("Sentiment test accuracy (%)")
    axes[0].set_ylim(83.3, 94.4)
    handles = []
    for family in ("rnn", "lstm", "transformer"):
        for variant in ("S", "M", "e10"):
            handles.append(
                Line2D(
                    [],
                    [],
                    ls="",
                    marker=RANKED_MARKERS[variant],
                    ms=5.0,
                    mfc=RANKED_SHADES[family][variant],
                    mec=RANKED_SHADES[family]["S"] if variant == "e10" else "white",
                    mew=0.7,
                    label=FAMILY[family] + RANKED_SUFFIX[variant],
                )
            )
    tables = [
        Line2D(
            [],
            [],
            ls="",
            marker="D",
            ms=4.6,
            mfc=PART2[key]["colour"],
            mec="white",
            mew=0.7,
            label=PART2[key]["label"],
        )
        for key in ("own_frozen", "glove_frozen", "glove_trainable", "random_frozen")
    ]
    # One legend column at the right: the nine Part I models, then the four tables.
    handles += tables
    fig.legend(
        handles=handles,
        loc="center right",
        ncol=1,
        fontsize=6.0,
        handletextpad=0.2,
        labelspacing=0.35,
        borderaxespad=0.1,
        bbox_to_anchor=(1.0, 0.5),
    )
    fig.tight_layout(rect=(0, 0, 0.8, 1), w_pad=1.0)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)


def figure_part3_length(out: Path) -> None:
    """Appendix figure: accuracy by review length, last word against average (D0.8)."""
    part3 = load(RESULTS / "part3/part3.json")
    reps = part3["representations"]
    groups = list(reps["rnn-S-full/pre_output/last"]["test"]["by_length_group"])
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 2.2), sharey=True)
    for ax, runs, title in ((axes[0], PART1_S, "(a) Size S"), (axes[1], PART1_M, "(b) Size M")):
        for run in runs:
            colour = MODEL_COLOURS[family_of(run)]
            for pooling, style in (("last", "-"), ("mean", TRAINING_SAMPLE_STYLE)):
                rows = reps[f"{run}/pre_output/{pooling}"]["test"]["by_length_group"]
                acc = [rows[g]["accuracy"] * 100 for g in groups]
                ax.plot(range(len(groups)), acc, color=colour, lw=1.2, ls=style, marker="o", ms=2.3)
        counts = reps["word_counts"]["test"]["by_length_group"]
        ax.plot(
            range(len(groups)),
            [counts[g]["accuracy"] * 100 for g in groups],
            color=INK_MUTED,
            lw=1.0,
        )
        ax.set_xticks(range(len(groups)), [f"{g.replace('-', chr(8211))} words" for g in groups])
        ax.set_title(title)
    axes[0].set_ylabel("Test accuracy (%)")
    handles = [
        Line2D([], [], color=MODEL_COLOURS[f], lw=1.2, label=FAMILY[f])
        for f in ("rnn", "lstm", "transformer")
    ]
    handles += [
        Line2D([], [], color=INK_SECONDARY, lw=1.2, label="last word"),
        Line2D([], [], color=INK_SECONDARY, lw=1.2, ls=TRAINING_SAMPLE_STYLE, label="average"),
        Line2D([], [], color=INK_MUTED, lw=1.0, label="word counts"),
    ]
    axes[1].legend(handles=handles, loc="lower left", fontsize=6.2, ncol=2)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- tables


def write(out: Path, lines: list[str]) -> None:
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def signed(value: float, digits: int = 3) -> str:
    """A number with a typeset minus sign when negative."""
    text = f"{value:.{digits}f}"
    return text.replace("-", "$-$") if value < 0 else text


def signed_points(share: float) -> str:
    """A difference of shares in percentage points, signed, never printed as -0.0."""
    value = round(100 * share, 1)
    if value == 0:
        return "0.0"
    return f"{value:+.1f}".replace("-", "$-$")


def rate(value: float) -> str:
    exponent = math.floor(math.log10(value) + 1e-9)
    mantissa = value / 10**exponent
    if abs(mantissa - 1) < 1e-6:
        return f"$10^{{{exponent}}}$"
    return f"${mantissa:.0f}{{\\times}}10^{{{exponent}}}$"


def table_part1(out: Path, facts: dict[str, Any]) -> None:
    """Body table: Part I quality, tuning and cost (D1.7)."""
    test = load(RESULTS / "eval/test.json")["models"]
    costs = load(RESULTS / "eval/costs.json")
    ngram = load(RESULTS / "ngram/ngram.json")
    rows = []
    kn_cost = costs["models"]["kneser-ney"]
    order = ngram["chosen"]["kneser_ney"]["order"]
    add_k = next(r for r in ngram["validation"] if r["method"] == "add_k" and r["order"] == order)
    for name in ("kneser-ney", "add-k"):
        model = test[name]
        cpu = (
            f"{kn_cost['speed']['cpu']['targets_per_second'] / 1000:,.0f}"
            if name == "kneser-ney"
            else "--"
        )
        time = (
            f"{kn_cost['counting_seconds'] / 60:.2f}"
            if name == "kneser-ney"
            else f"{(kn_cost['counting_seconds'] + add_k['search_seconds']) / 60:.2f}"
        )
        rows.append(
            [
                SHORT[name],
                f"{model['test']['J']:.3f}",
                f"{model['test']['PPL']:.1f}",
                f"{model['valid_J']:.3f}",
                "--" if name == "kneser-ney" else f"$k{{=}}10^{{{round(math.log10(add_k['k']))}}}$",
                "--",
                "--",
                "--",
                cpu,
                time,
                "--",
                f"{kn_cost['table_bytes'] / 1e6:.0f}",
            ]
        )
    groups = [
        ("Size S (main)", PART1_S),
        ("Size M (size check)", PART1_M),
        ("Size S, 10 epochs (budget check)", PART1_E10),
    ]
    body: list[list[str] | str] = [*rows]
    for title, runs in groups:
        body.append(title)
        for run in runs:
            model = test[run]
            base = run.replace("-e10", "")
            info = costs["models"].get(base, {})
            training = costs["training"][run]
            tuning_key = base.replace("-full", "")
            tuning = costs["tuning"].get(tuning_key)
            peak = run_record(run)["summary"]["config"]["peak_lr"]
            gpu = info.get("speed", {}).get("gpu")
            cpu = info.get("speed", {}).get("cpu")
            e10 = "-e10-" in run
            body.append(
                [
                    FAMILY[family_of(run)] + ("-M" if "-M-" in run else "-S"),
                    f"{model['test']['J']:.3f}",
                    f"{model['test']['PPL']:.1f}",
                    f"{model['valid_J']:.3f}",
                    rate(peak) + ("" if e10 else f" ({tuning['count']})"),
                    "" if e10 else n(info["parameters"]["core"]),
                    "" if e10 else f"{info['flops_per_prediction']['total'] / 1e6:.2f}",
                    "" if e10 or gpu is None else f"{gpu['targets_per_second'] / 1000:.0f}",
                    "" if e10 or cpu is None else f"{cpu['targets_per_second'] / 1000:.1f}",
                    f"{training['train_seconds'] / 60:.1f}",
                    "" if e10 else f"{tuning['total_seconds'] / 3600:.2f}",
                    "" if e10 else f"{info['checkpoint_bytes'] / 1e6:.1f}",
                ]
            )
    lines = [
        r"\begin{tabular}{@{}lrrlrrrrrrr@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Test} & & \multicolumn{3}{c}{Size and arithmetic} & "
        r"\multicolumn{2}{c}{K targets/s} & \multicolumn{2}{c}{Training} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){5-7}\cmidrule(lr){8-9}\cmidrule(l){10-11}",
        r"Model & $J$ & PPL & Peak LR (tries) & Core & MFLOP & MB & GPU & CPU & min & tries (h) \\",
        r"\midrule",
    ]
    for row in body:
        if isinstance(row, str):
            lines.append(r"\addlinespace[3pt]")
            lines.append(rf"\multicolumn{{11}}{{@{{}}l}}{{\textbf{{{row}}}}} \\")
            continue
        name, j, ppl, vj, peak, core, flops, gpu, cpu, final, tries, mb = row
        lines.append(
            " & ".join([name, j, ppl, peak, core, flops, mb, gpu, cpu, final, tries]) + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, lines)
    facts["part1"] = {
        run: {
            "test_J": test[run]["test"]["J"],
            "PPL": test[run]["test"]["PPL"],
            "valid_J": test[run]["valid_J"],
        }
        for run in ("kneser-ney", "add-k", *PART1_S, *PART1_M, *PART1_E10)
    }


def table_tuning(out: Path) -> None:
    """Appendix table: every learning-rate try (validation J of the best checkpoint)."""
    rates = [0.0003, 0.001, 0.003, 0.01, 0.03]
    lines = [
        r"\begin{tabular}{@{}l" + "r" * len(rates) + "@{}}",
        r"\toprule",
        "Model & " + " & ".join(rate(r) for r in rates) + r" \\",
        r"\midrule",
    ]
    for model in ("rnn-S", "lstm-S", "transformer-S", "rnn-M", "lstm-M", "transformer-M"):
        tuning = load(RESULTS / "runs" / f"{model}-tuning.json")
        tries = {float(k): v for k, v in tuning["tries"].items()}
        best = min(tries.values())
        cells = []
        for r in rates:
            value = next((v for k, v in tries.items() if math.isclose(k, r)), None)
            if value is None:
                cells.append("")
            elif value == best:
                cells.append(rf"\textbf{{{value:.3f}}}")
            else:
                cells.append(f"{value:.3f}")
        family, size = model.split("-")
        assert math.isclose(tuning["chosen"], min(tries, key=tries.get)), model
        lines.append(f"{FAMILY[family]}-{size} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, lines)


def table_positions(out: Path, facts: dict[str, Any]) -> None:
    """Appendix table: test J by target position (D1.5), with Kneser-Ney's own J."""
    test = load(RESULTS / "eval/test.json")["models"]
    kn = test["kneser-ney"]["by_position"]
    groups = [f"{g['positions'][0]}-{g['positions'][1]}" for g in kn]
    lines = [
        r"\begin{tabular}{@{}l" + "r" * len(groups) + "@{}}",
        r"\toprule",
        "Targets at position & " + " & ".join(g.replace("-", "--") for g in groups) + r" \\",
        r"\midrule",
        "Test targets & " + " & ".join(n(g["targets"]) for g in kn) + r" \\",
        "Kneser-Ney $J$ & " + " & ".join(f"{g['J']:.3f}" for g in kn) + r" \\",
        r"\midrule",
        r"\multicolumn{"
        + str(len(groups) + 1)
        + r"}{@{}l}{\emph{Lead over Kneser-Ney (its $J$ minus the model's)}} \\",
    ]
    leads = {}
    for run in (*PART1_S, *PART1_M, *PART1_E10):
        rows = test[run]["by_position"]
        lead = [k["J"] - r["J"] for k, r in zip(kn, rows, strict=True)]
        leads[run] = lead
        name = (
            FAMILY[family_of(run)]
            + ("-M" if "-M-" in run else "-S")
            + (", 10 ep." if "e10" in run else "")
        )
        lines.append(f"{name} & " + " & ".join(signed(v) for v in lead) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, lines)
    facts["positions"] = {"groups": groups, "kn_J": [g["J"] for g in kn], "lead": leads}


def table_generation(out: Path, facts: dict[str, Any]) -> None:
    """Body table: scores of generated text against real continuations (D1.6)."""
    generation = load(RESULTS / "eval/generation.json")
    scores = generation["scores"]
    names = {
        "real": "Real continuation",
        "kneser-ney": "Kneser-Ney 5-gram",
        "rnn-S-full": "RNN-S",
        "lstm-S-full": "LSTM-S",
        "transformer-S-full": "Transformer-S",
        "lstm-own-frozen-s4213-full": "Own skip-gram, frozen",
        "lstm-glove-frozen-s4213-full": "GloVe, frozen",
        "lstm-glove-trainable-s4213-full": "GloVe, trainable",
        "lstm-random-frozen-s4213-full": "Random, frozen",
    }
    lines = [
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Text & Rep-4 & Dist-2 & Copied & Gap \\",
        r"\midrule",
    ]
    for key, label in names.items():
        if key == "lstm-own-frozen-s4213-full":
            lines.append(r"\midrule")
        s = scores[key]["scores"]
        gap = scores[key].get("gap")
        lines.append(
            f"{label} & {100 * s['seq_rep_4']:.1f}\\% & {s['distinct_2']:.3f} & "
            f"{100 * s['copied_5gram_share']:.1f}\\% & "
            + ("" if gap is None else f"{gap:+.2f}".replace("-", "$-$"))
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, lines)
    facts["generation"] = {
        "temperature": generation["temperature"],
        "choice": generation["temperature_test"]["choice"],
        "scores": {k: {**scores[k]["scores"], "gap": scores[k].get("gap")} for k in names},
        "gap_at_0.8": {
            run: generation["temperature_test"]["measured"][run]["0.8"]["gap"]
            for run in generation["temperature_test"]["measured"]
        },
        "rep4_at_0.8": {
            run: generation["temperature_test"]["measured"][run]["0.8"]["seq_rep_4"]
            for run in generation["temperature_test"]["measured"]
        },
        "dist2_at_0.8": {
            run: generation["temperature_test"]["measured"][run]["0.8"]["distinct_2"]
            for run in generation["temperature_test"]["measured"]
        },
    }


def middle_samples(path: Path) -> dict[tuple[int, str], str]:
    """The sample marked as the median of nine (">>") for every prompt and model."""
    found: dict[tuple[int, str], str] = {}
    header = re.compile(r"^\[prompt (\d+): [^\]]*\] \[([^\]]+)\]")
    prompt, model = None, None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = header.match(line)
        if match:
            prompt, model = int(match.group(1)), match.group(2)
            continue
        if line.startswith(">>") and prompt is not None:
            text = re.sub(r"^>>\s+ppl\s+[\d.]+\s+", "", line)
            found[(prompt, model)] = text
    return found


def table_samples(out: Path) -> bool:
    """Body table: the median-of-nine sample for prompts 3 and 6 (D1.6)."""
    path = RESULTS / "eval/examples_generation.txt"
    if not path.exists():
        return False
    generation = load(RESULTS / "eval/generation.json")
    prompts = generation["report_body_prompts"]
    texts = {p: detokenize(generation["prompts"][str(p)]["text"].split()) for p in prompts}
    samples = middle_samples(path)
    models = [
        ("kneser-ney", "KN"),
        ("rnn-S-full", "RNN"),
        ("lstm-S-full", "LSTM"),
        ("transformer-S-full", "Trans."),
        ("lstm-own-frozen-s4213-full", "Own fr."),
        ("lstm-glove-frozen-s4213-full", "GloVe fr."),
    ]
    width = r">{\raggedright\arraybackslash}p{0.43\textwidth}"
    lines = [
        r"\begin{tabular}{@{}l" + width + width + "@{}}",
        r"\toprule",
        " & ".join(["", *[r"\emph{" + tex_escape(texts[p]) + r"} \ldots" for p in prompts]])
        + r" \\",
        r"\midrule",
    ]
    for model, label in models:
        if model == "lstm-own-frozen-s4213-full":
            lines.append(r"\midrule")
        cells = []
        for p in prompts:
            text = samples[(p, model)]
            prefix = texts[p]
            assert text.startswith(prefix), (p, model)
            cells.append(tex_escape(text[len(prefix) :].strip()))
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, lines)
    return True


def table_samples_all(out: Path) -> bool:
    """Appendix table: the median-of-nine sample of all eight models for all six prompts (D1.6)."""
    path = RESULTS / "eval/examples_generation.txt"
    if not path.exists():
        return False
    generation = load(RESULTS / "eval/generation.json")
    prompts = sorted(int(p) for p in generation["prompts"])
    texts = {p: detokenize(generation["prompts"][str(p)]["text"].split()) for p in prompts}
    samples = middle_samples(path)
    models = [
        ("kneser-ney", "KN 5-gram"),
        ("rnn-S-full", "RNN-S"),
        ("lstm-S-full", "LSTM-S"),
        ("transformer-S-full", "Transformer-S"),
        ("lstm-own-frozen-s4213-full", "Own, frozen"),
        ("lstm-glove-frozen-s4213-full", "GloVe, frozen"),
        ("lstm-glove-trainable-s4213-full", "GloVe, trainable"),
        ("lstm-random-frozen-s4213-full", "Random, frozen"),
    ]
    lines = [
        r"\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{0.82\textwidth}@{}}",
        r"\toprule",
    ]
    for number, p in enumerate(prompts):
        prompt = texts[p]
        shown = (
            r"\emph{" + tex_escape(prompt) + r"} \ldots"
            if prompt
            else r"(no prompt: the start of a review)"
        )
        if number:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{2}}{{@{{}}l}}{{Prompt {p}: {shown}}} \\")
        lines.append(r"\addlinespace[1pt]")
        for model, label in models:
            text = samples[(p, model)]
            assert text.startswith(prompt), (p, model)
            lines.append(f"{label} & " + tex_escape(text[len(prompt) :].strip()) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, lines)
    return True


def table_part2(out: Path, facts: dict[str, Any]) -> None:
    """Body table: Part II test J per seed, convergence and stability (P2.7)."""
    test = load(RESULTS / "eval/test.json")["models"]
    lines = [
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Test $J$} & \multicolumn{2}{c}{Valid $J$, epoch} & \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Input table & Mean & SD & 1 & 6 & V$-$T \\",
        r"\midrule",
    ]
    out_facts: dict[str, Any] = {}
    for key, setting in PART2.items():
        runs = setting["runs"]
        js = [test[r]["test"]["J"] for r in runs]
        epoch_ends = [epoch_end_valid(r) for r in runs]
        first = np.mean([e[0] for e in epoch_ends])
        last = np.mean([e[-1] for e in epoch_ends])
        gaps = []
        clipped = []
        for r in runs:
            x, valid, train = run_checks(r)
            best = int(np.argmin(valid))
            gaps.append(valid[best] - train[best])
            clipped.append(run_record(r)["summary"]["gradient_norms"]["share_clipped"])
        sd = float(np.std(js, ddof=1)) if len(js) > 1 else None
        lines.append(
            f"{setting['short']} & {np.mean(js):.3f} & "
            + ("--" if sd is None else f"{sd:.3f}")
            + f" & {first:.3f} & {last:.3f} & {np.mean(gaps):.3f} \\\\"
        )
        out_facts[key] = {
            "runs": list(runs),
            "test_J": js,
            "mean": float(np.mean(js)),
            "sd": sd,
            "epoch_end_valid": epoch_ends,
            "gap": [float(g) for g in gaps],
            "clipped": clipped,
        }
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, lines)
    random_runs = PART2["random_trainable"]["runs"]
    paired = {}
    for other in ("own_frozen", "glove_frozen"):
        paired[other] = [
            test[o]["test"]["J"] - test[r]["test"]["J"]
            for o, r in zip(PART2[other]["runs"], random_runs, strict=True)
        ]
    paired["own_minus_glove"] = [
        test[o]["test"]["J"] - test[g]["test"]["J"]
        for o, g in zip(PART2["own_frozen"]["runs"], PART2["glove_frozen"]["runs"], strict=True)
    ]
    j = {key: test[s["runs"][0]]["test"]["J"] for key, s in PART2.items()}
    out_facts["paired"] = paired
    out_facts["decomposition_seed_4213"] = {
        "freezing_cost": j["random_frozen"] - j["random_trainable"],
        "own_wins_back": j["random_frozen"] - j["own_frozen"],
        "glove_wins_back": j["random_frozen"] - j["glove_frozen"],
        "freezing_glove_costs": j["glove_frozen"] - j["glove_trainable"],
        "glove_start_vs_random": j["glove_trainable"] - j["random_trainable"],
    }
    facts["part2"] = out_facts


def rep_acc(reps: dict[str, Any], key: str) -> str:
    return f"{100 * reps[key]['test']['accuracy']:.1f}"


def table_part3(out: Path, facts: dict[str, Any]) -> None:
    """Body table: test accuracy of every representation (D3.2, D3.3)."""
    part3 = load(RESULTS / "part3/part3.json")
    reps = part3["representations"]
    costs = part3["costs"]
    groups = [
        ("Design (S)", PART1_S),
        ("Size (M)", PART1_M),
        ("Training length (S, 10 epochs)", PART1_E10),
        (
            "Word table (LSTM-S)",
            (
                "lstm-own-frozen-s4213-full",
                "lstm-glove-frozen-s4213-full",
                "lstm-glove-trainable-s4213-full",
                "lstm-random-frozen-s4213-full",
            ),
        ),
    ]
    lines = [
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Representation & $J$ & Last & Avg. \\",
        r"\midrule",
    ]
    for title, runs in groups:
        lines.append(rf"\multicolumn{{4}}{{@{{}}l}}{{\emph{{{title}}}}} \\")
        for run in runs:
            name = SHORT[run].replace(", 10 ep.", "").replace("Own skip-gram", "Own")
            lines.append(
                f"\\quad {name} & {costs[run]['test_J']:.3f} & "
                f"{rep_acc(reps, run + '/pre_output/last')} & "
                f"{rep_acc(reps, run + '/pre_output/mean')} \\\\"
            )
    lines.append(r"\multicolumn{4}{@{}l}{\emph{References}} \\")

    def span(keys: list[str]) -> str:
        values = sorted(100 * reps[k]["test"]["accuracy"] for k in keys)
        return f"{values[0]:.1f}--{values[-1]:.1f}"

    untrained = [f"untrained/{f}-S-full/pre_output" for f in ("rnn", "lstm", "transformer")]
    lines.append(
        f"\\quad Untrained (S) & -- & {span([u + '/last' for u in untrained])} & "
        f"{span([u + '/mean' for u in untrained])} \\\\"
    )
    lines.append(
        "\\quad Tables, averaged & -- & & "
        f"{span(['table/learned', 'table/skipgram', 'table/glove'])} \\\\"
    )
    width = reps["word_counts"]["width"]
    lines.append(
        f"\\quad Word counts ({width / 1000:.0f}K col.) & -- & "
        f"\\multicolumn{{2}}{{c}}{{{rep_acc(reps, 'word_counts')}}} \\\\"
    )
    lines.append(
        f"\\quad Length only & -- & \\multicolumn{{2}}{{c}}{{{rep_acc(reps, 'length')}}} \\\\"
    )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, lines)

    intervals = [
        reps[k]["test"]["interval"][1] - reps[k]["test"]["interval"][0]
        for k in reps
        if "test" in reps[k]
    ]
    facts["part3"] = {
        "accuracy": {k: reps[k]["test"]["accuracy"] for k in reps},
        "interval": {k: reps[k]["test"]["interval"] for k in reps},
        "interval_width_range": [min(intervals), max(intervals)],
        "C": {k: reps[k]["C"] for k in reps},
        "ranking": part3["ranking"],
        "luck": part3["luck"],
        "data": part3["data"],
        "word_count_columns": width,
    }


POOL_NAMES = {"last": "last word", "mean": "average", "last vs mean": "last vs average", "": ""}
PART2_NAMES = {
    "own-frozen": "Own, frozen",
    "glove-frozen": "GloVe, frozen",
    "glove-trainable": "GloVe, trainable",
    "random-frozen": "Random, frozen",
    "random": "Random, trainable",
}
TABLE_NAMES = {
    "learned": "Averaged table, LSTM-S's own",
    "skipgram": "Averaged table, skip-gram",
    "glove": "Averaged table, GloVe",
}


def display_name(key: str) -> str:
    """The report's name for a Part III representation key (no pooling)."""
    parts = key.split("/")
    untrained = parts[0] == "untrained"
    if untrained:
        parts = parts[1:]
    if parts[0] == "table":
        return TABLE_NAMES[parts[1]]
    if parts[0] == "word_counts":
        return "Word counts (tf-idf)"
    if parts[0] == "length":
        return "Length only"
    run = parts[0].replace("-full", "")
    match = re.match(
        r"lstm-(own-frozen|glove-frozen|glove-trainable|random-frozen|random)-s(\d+)$", run
    )
    if match:
        name = PART2_NAMES[match.group(1)]
        if match.group(2) != "4213":
            name += f" (seed {match.group(2)})"
    else:
        family, size = run.split("-", 1)
        name = FAMILY[family] + "-" + size.replace("-e10", ", 10 ep.")
    if untrained:
        name = "Untrained " + name
    if len(parts) > 1 and parts[1] == "hidden":
        name += " (top hidden)"
    return name


def table_part3_all(out: Path) -> None:
    """Appendix table: every representation's chosen C, cross-validation and test accuracy."""
    part3 = load(RESULTS / "part3/part3.json")
    reps = part3["representations"]
    lines = [
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r"Representation & Pool & $C$ & CV acc. & Test acc. & 95\% interval & Macro-F1 \\",
        r"\midrule",
    ]
    for key, rep in reps.items():
        parts = key.split("/")
        pooling = POOL_NAMES[parts[-1] if parts[-1] in ("last", "mean") else ""]
        name = display_name(key)
        cv = rep["mean_accuracy"][str(rep["power"])]
        t = rep["test"]
        lo, hi = t["interval"]
        lines.append(
            f"{tex_escape(name)} & {pooling} & $10^{{{rep['power']}}}$ & {100 * cv:.1f} & "
            f"{100 * t['accuracy']:.1f} & {100 * lo:.1f}--{100 * hi:.1f} & "
            f"{100 * t['macro_f1']:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, lines)


def table_comparisons(out: Path, facts: dict[str, Any]) -> None:
    """Appendix table: the 51 comparisons fixed in advance (exact McNemar, Holm)."""
    part3 = load(RESULTS / "part3/part3.json")
    rows = part3["comparisons"]

    def short(key: str) -> str:
        return tex_escape(display_name(key))

    lines = [
        r"\begin{tabular}{@{}llrrrrl@{}}",
        r"\toprule",
        r"First & Second & Pool & Diff. (pts) & 95\% interval & Holm $p$ & Real? \\",
        r"\midrule",
    ]
    significant = 0
    robust = 0
    for c in rows:
        lo, hi = c["difference_interval"]
        sig = c["significant_at_0.05"]
        product = c["product_interval_excludes_0"]
        significant += int(sig)
        robust += int(sig and product)
        p = c["p_holm"]
        p_text = "$<$0.001" if p < 0.001 else f"{p:.3f}"
        lines.append(
            f"{short(c['first'])} & {short(c['second'])} & {POOL_NAMES[c['pooling']]} & "
            f"{signed_points(c['difference'])} & {signed_points(lo)} to {signed_points(hi)} & "
            f"{p_text} & " + ("yes" if sig and product else ("Holm only" if sig else "no")) + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, lines.copy())
    facts["comparisons"] = {
        "count": len(rows),
        "significant": significant,
        "significant_and_product_robust": robust,
        "rows": rows,
    }


def category_table(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def table_errors(out: Path, facts: dict[str, Any]) -> None:
    """Body table: accuracy in measured groups and categories of 40 random mistakes (D3.4)."""
    errors = load(RESULTS / "part3/errors.json")
    main, reference = errors["main"], errors["reference"]
    acc = errors["group_accuracy"]
    groups = [
        ("all", "All test reviews"),
        ("4 stars", "4 stars"),
        ("contrast", "Contrast word (but, though, \\ldots)"),
        ("no contrast", "No contrast word"),
        ("negation", "Negation word (not, n't, \\ldots)"),
    ]
    lines = [
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Test reviews & $n$ & Main & Counts \\",
        r"\midrule",
    ]
    for key, label in groups:
        row_main = acc[main][key]
        row_ref = acc[reference][key]
        lines.append(
            f"{label} & {n(row_main['reviews'])} & {100 * row_main['accuracy']:.1f} & "
            f"{100 * row_ref['accuracy']:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, lines)
    # The categories of 40 random mistakes go to an appendix table of their own.
    rows = category_table(RESULTS / "part3/error_categories.csv")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    total = len(rows)
    categories = [
        r"\begin{tabular}{@{}lrr@{}}",
        r"\toprule",
        rf"Category ({total} random mistakes) & $n$ & 95\% interval \\",
        r"\midrule",
    ]
    for category, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lo, hi = wilson(count, total)
        label = category[0].upper() + category[1:]
        categories.append(f"{tex_escape(label)} & {count} & {100 * lo:.0f}--{100 * hi:.0f}\\% \\\\")
    categories += [r"\bottomrule", r"\end{tabular}"]
    write(out.with_name("error_categories.tex"), categories)
    facts["errors"] = {
        "main": main,
        "reference": reference,
        "group_accuracy": acc,
        "together": errors["main_against_reference"],
        "ranking_in_groups": errors["ranking_in_groups"],
        "categories": {k: {"count": v, "wilson": wilson(v, total)} for k, v in counts.items()},
        "category_total": total,
    }


def table_part2_words(out: Path, facts: dict[str, Any]) -> None:
    """Appendix table: coverage, word tests and neighbours of the Part II tables."""
    emb = load(RESULTS / "part2/embeddings.json")
    cov = emb["glove"]["coverage"]
    intrinsic = emb["intrinsic"]
    neighbours = emb["neighbours"]
    lines = [
        r"\begin{tabularx}{\textwidth}{@{}l>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X@{}}",
        r"\toprule",
        r" & Own skip-gram (10.7M training tokens of reviews) & "
        r"GloVe 2024 (11.9B tokens of Wikipedia and Gigaword) \\",
        r"\midrule",
        f"WordSim353 Spearman ({intrinsic['glove']['wordsim353_pairs']} of "
        f"{intrinsic['glove']['wordsim353_pairs_total']} pairs) & "
        f"{intrinsic['skipgram']['wordsim353_spearman']:.2f} & "
        f"{intrinsic['glove']['wordsim353_spearman']:.2f} \\\\",
        f"Analogies correct ({n(intrinsic['glove']['analogy_questions'])} of "
        f"{n(intrinsic['glove']['analogy_questions_total'])}) & "
        f"{100 * intrinsic['skipgram']['analogy_accuracy']:.1f}\\% & "
        f"{100 * intrinsic['glove']['analogy_accuracy']:.1f}\\% \\\\",
    ]
    for word in ("cheap", "broke", "gift", "great"):
        lines.append(
            f"Neighbours of ``{word}'' & {', '.join(neighbours['skipgram'][word])} & "
            f"{', '.join(neighbours['glove'][word])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabularx}"]
    write(out, lines)
    facts["part2_words"] = {
        "coverage": cov,
        "intrinsic": intrinsic,
        "skipgram_seconds": emb["skipgram"]["seconds"],
        "skipgram_tokens_processed": emb["skipgram"]["tokens_processed"],
        "scale": emb["scale"],
        "norm_check": emb["norm_check"],
    }


def exploratory_part3(out: Path, facts: dict[str, Any]) -> bool:
    """Appendix table: checks added while writing, not fixed in advance (labelled exploratory).

    Ranking agreement by the size of the J gap, the ranking without the RNNs, and
    exact McNemar tests of the M models against word counts (no correction).
    Needs data/part3/predictions.npz (git-ignored) for the McNemar rows.
    """
    import itertools

    from scipy.stats import kendalltau

    from a1.part3.stats import mcnemar

    part3 = load(RESULTS / "part3/part3.json")
    ranking = part3["ranking"]
    rows: dict[str, Any] = {"pairs_by_gap": {}, "without_rnn": {}, "discordant": {}}
    bins = ((0.0, 0.05), (0.05, 0.15), (0.15, math.inf))
    for pooling in ("last", "mean"):
        models = ranking[pooling]["models"]
        counts = {f"{lo}-{hi}": [0, 0] for lo, hi in bins}
        discordant = []
        for a, b in itertools.combinations(models, 2):
            gap = abs(a["test_J"] - b["test_J"])
            agree = (a["test_J"] - b["test_J"]) * (a["accuracy"] - b["accuracy"]) < 0
            for lo, hi in bins:
                if lo <= gap < hi:
                    counts[f"{lo}-{hi}"][0] += int(agree)
                    counts[f"{lo}-{hi}"][1] += 1
            if not agree:
                discordant.append(
                    {
                        "pair": [a["model"], b["model"]],
                        "J_gap": gap,
                        "accuracy_gap": abs(a["accuracy"] - b["accuracy"]),
                    }
                )
        rows["pairs_by_gap"][pooling] = counts
        rows["discordant"][pooling] = discordant
        kept = [m for m in models if not m["model"].startswith("rnn")]
        tau = kendalltau([m["test_J"] for m in kept], [m["accuracy"] for m in kept])
        same = sum(
            1
            for a, b in itertools.combinations(kept, 2)
            if (a["test_J"] - b["test_J"]) * (a["accuracy"] - b["accuracy"]) < 0
        )
        rows["without_rnn"][pooling] = {
            "models": len(kept),
            "tau": -float(tau.statistic),
            "p": float(tau.pvalue),
            "same_order": same,
            "pairs": len(kept) * (len(kept) - 1) // 2,
        }
    predictions = Path("data/part3/predictions.npz")
    tests = {}
    if predictions.exists():
        with np.load(predictions, allow_pickle=False) as stored:
            names = stored["names"].tolist()
            predicted = stored["predicted"]
            labels = stored["labels"]
        right = {name: predicted[i] == labels for i, name in enumerate(names)}
        for name in (
            "lstm-M-full/pre_output/last",
            "transformer-M-full/pre_output/last",
            "lstm-M-full/hidden/last",
            "transformer-M-full/hidden/last",
        ):
            tests[name] = mcnemar(right[name], right["word_counts"])
        main = "transformer-M-full/pre_output/last"
        rows["union_main_counts"] = float((right[main] | right["word_counts"]).mean())
    rows["mcnemar_against_word_counts"] = tests
    facts["exploratory"] = rows
    lines = [
        r"\begin{tabular}{@{}lrr@{}}",
        r"\toprule",
        r"Pairs in the same order & Last word & Average \\",
        r"\midrule",
    ]
    for lo, hi in bins:
        key = f"{lo}-{hi}"
        label = f"$J$ gap {lo:g} to {hi:g}" if math.isfinite(hi) else f"$J$ gap {lo:g} or more"
        cells = [
            f"{rows['pairs_by_gap'][p][key][0]} of {rows['pairs_by_gap'][p][key][1]}"
            for p in ("last", "mean")
        ]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    cells = [
        f"{rows['without_rnn'][p]['tau']:.2f} ({rows['without_rnn'][p]['same_order']} of "
        f"{rows['without_rnn'][p]['pairs']})"
        for p in ("last", "mean")
    ]
    lines.append(r"\midrule")
    lines.append(r"$\tau$ without the RNNs & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, lines)
    return bool(tests)


def derived_facts(facts: dict[str, Any]) -> dict[str, Any]:
    """Every derived number the report text quotes, computed here from the result files.

    scripts/check_report_numbers.py checks the numbers in the text against these.
    """
    test = load(RESULTS / "eval/test.json")["models"]
    costs = load(RESULTS / "eval/costs.json")
    generation = load(RESULTS / "eval/generation.json")
    part3 = load(RESULTS / "part3/part3.json")
    errors = load(RESULTS / "part3/errors.json")
    emb = load(RESULTS / "part2/embeddings.json")
    reps = part3["representations"]
    j = {name: model["test"]["J"] for name, model in test.items()}
    acc = {key: 100 * rep["test"]["accuracy"] for key, rep in reps.items()}
    d: dict[str, Any] = {}
    neural = [name for name in test if test[name]["family"] != "ngram"]
    d["neural_models"] = len(neural)
    d["test_J_range"] = [min(j[m] for m in neural), max(j[m] for m in neural)]
    d["kn_lead_S"] = {run: j["kneser-ney"] - j[run] for run in PART1_S}
    d["transformer_minus_lstm_S"] = j["transformer-S-full"] - j["lstm-S-full"]
    d["lstm_minus_transformer_M"] = j["lstm-M-full"] - j["transformer-M-full"]
    d["transformer_minus_lstm_e10"] = j["transformer-S-e10-full"] - j["lstm-S-e10-full"]
    seed_sd = facts["part2"]["random_trainable"]["sd"]
    d["lstm_transformer_gap_in_difference_sds"] = d["transformer_minus_lstm_S"] / (
        math.sqrt(2) * seed_sd
    )
    lead = facts["positions"]["lead"]
    d["transformer_minus_lstm_by_position_S"] = [
        a - b for a, b in zip(lead["lstm-S-full"], lead["transformer-S-full"], strict=True)
    ]
    kn = test["kneser-ney"]["by_position"]
    total = sum(g["targets"] for g in kn)
    d["share_targets_from_65"] = sum(g["targets"] for g in kn if g["positions"][0] >= 65) / total
    d["share_targets_from_129"] = sum(g["targets"] for g in kn if g["positions"][0] >= 129) / total
    gap_rows = [
        (g["targets"] / total) * (t["J"] - lstm["J"])
        for g, lstm, t in zip(
            kn,
            test["lstm-S-full"]["by_position"],
            test["transformer-S-full"]["by_position"],
            strict=True,
        )
    ]
    d["gap_from_65_on"] = sum(
        v for g, v in zip(kn, gap_rows, strict=True) if g["positions"][0] >= 65
    )
    models = costs["models"]
    flops = {run: models[run]["flops_per_prediction"] for run in (*PART1_S, *PART1_M)}
    d["output_layer_share_S"] = [flops[r]["output_layer"] / flops[r]["total"] for r in PART1_S]
    d["M_over_S_flops"] = [
        flops[m]["total"] / flops[s]["total"] for s, m in zip(PART1_S, PART1_M, strict=True)
    ]
    d["M_over_S_core"] = [
        models[m]["parameters"]["core"] / models[s]["parameters"]["core"]
        for s, m in zip(PART1_S, PART1_M, strict=True)
    ]
    train = {run: costs["training"][run]["train_seconds"] for run in costs["training"]}
    d["M_over_S_training_time_percent_more"] = [
        100 * (train[m] / train[s] - 1) for s, m in zip(PART1_S, PART1_M, strict=True)
    ]
    d["e10_over_S_training_time"] = [
        train[e] / train[s] for s, e in zip(PART1_S, PART1_E10, strict=True)
    ]
    kn_speed = models["kneser-ney"]["speed"]["cpu"]["targets_per_second"]
    d["kn_cpu_speedup"] = [
        kn_speed / models[r]["speed"]["cpu"]["targets_per_second"] for r in PART1_S
    ]
    gaps, drops = {}, {}
    for run in (*PART1_S, *PART1_M, *PART1_E10):
        x, valid, train_j = run_checks(run)
        best = int(np.argmin(valid))
        gaps[run] = float(valid[best] - train_j[best])
        ends = epoch_end_valid(run)
        drops[run] = ends[-2] - ends[-1]
    d["train_valid_gap_at_best"] = gaps
    d["last_epoch_drop"] = drops
    d["checks_per_6_epoch_run"] = len(run_checks("lstm-S-full")[0])
    d["glove_tokens_over_ours"] = 11.9e9 / emb["lm_train"]["tokens"]
    d["glove_tokens_over_ours_rounded"] = round(d["glove_tokens_over_ours"], -2)
    p2 = facts["part2"]
    d["part2_mean_cost"] = {
        key: p2[key]["mean"] - p2["random_trainable"]["mean"]
        for key in ("own_frozen", "glove_frozen")
    }
    # The same cost as a change in perplexity (exp of the J difference), for the abstract.
    d["part2_mean_cost_perplexity_percent"] = {
        key: (math.exp(cost) - 1) * 100 for key, cost in d["part2_mean_cost"].items()
    }
    end_gap = []
    for own, rnd in zip(
        PART2["own_frozen"]["runs"], PART2["random_trainable"]["runs"], strict=True
    ):
        end_gap.append(run_checks(own)[1][-1] - run_checks(rnd)[1][-1])
    d["own_minus_random_valid_at_end"] = end_gap
    first = []
    for own, rnd in zip(
        PART2["own_frozen"]["runs"], PART2["random_trainable"]["runs"], strict=True
    ):
        first.append([run_checks(own)[1][0], run_checks(rnd)[1][0]])
    d["own_and_random_first_check"] = first
    d["lstm_run_minutes"] = [train[r] / 60 for s in PART2.values() for r in s["runs"] if r in train]
    d["skipgram_minutes"] = emb["skipgram"]["seconds"] / 60
    norms = [
        run_record(r)["summary"]["gradient_norms"]["mean"]
        for s in PART2.values()
        for r in s["runs"]
    ]
    d["part2_gradient_norm_mean_range"] = [min(norms), max(norms)]
    half = [
        (rep["test"]["interval"][1] - rep["test"]["interval"][0]) * 50
        for key, rep in reps.items()
        if rep["kind"] in ("language_model", "hidden_check", "table_average", "word_counts")
    ]
    d["bootstrap_half_width_points"] = [min(half), max(half)]
    untrained = [acc[k] for k in acc if k.startswith("untrained/")]
    d["untrained_accuracy_range"] = [min(untrained), max(untrained)]
    d["training_gain_last"] = [
        acc[f"{f}-S-full/pre_output/last"] - acc[f"untrained/{f}-S-full/pre_output/last"]
        for f in ("rnn", "lstm", "transformer")
    ]
    d["lstm_over_own_table"] = acc["lstm-S-full/pre_output/last"] - acc["table/learned"]
    ranked = [m["model"] for m in part3["ranking"]["last"]["models"]]
    d["last_minus_mean"] = {
        m: acc[f"{m}/pre_output/last"] - acc[f"{m}/pre_output/mean"] for m in ranked
    }
    d["max_abs_last_minus_mean_non_rnn"] = max(
        abs(v) for m, v in d["last_minus_mean"].items() if not m.startswith("rnn")
    )
    discordant = facts["exploratory"]["discordant"]["last"]
    d["discordant_last"] = len(discordant)
    d["discordant_last_with_rnn"] = sum(
        any(x.startswith("rnn") for x in r["pair"]) for r in discordant
    )
    other = [
        100 * r["accuracy_gap"]
        for r in discordant
        if not any(x.startswith("rnn") for x in r["pair"])
    ]
    d["discordant_last_other_accuracy_gap_points"] = [min(other), max(other)]

    def change(a: str, b: str, pooling: str) -> float:
        return acc[f"{b}/pre_output/{pooling}"] - acc[f"{a}/pre_output/{pooling}"]

    d["size_effect"] = {
        pooling: {s: change(s, m, pooling) for s, m in zip(PART1_S, PART1_M, strict=True)}
        for pooling in ("last", "mean")
    }
    d["epochs_effect"] = {
        pooling: {s: change(s, e, pooling) for s, e in zip(PART1_S, PART1_E10, strict=True)}
        for pooling in ("last", "mean")
    }
    d["word_table_loss"] = {
        pooling: {
            r: change(r, "lstm-S-full", pooling)
            for r in (
                "lstm-own-frozen-s4213-full",
                "lstm-glove-frozen-s4213-full",
                "lstm-glove-trainable-s4213-full",
                "lstm-random-frozen-s4213-full",
            )
        }
        for pooling in ("last", "mean")
    }
    d["counts_minus_best"] = [
        acc["word_counts"] - acc[f"{m}/pre_output/last"]
        for m in ("lstm-M-full", "transformer-M-full")
    ]
    groups = errors["group_accuracy"][errors["main"]]
    d["contrast_share"] = groups["contrast"]["reviews"] / groups["all"]["reviews"]
    together = errors["main_against_reference"]
    d["one_side_right_mean"] = (together["first_only"] + together["second_only"]) / 2
    d["at_least_one_right"] = 100 * (1 - together["both_wrong"] / groups["all"]["reviews"])
    tau = part3["ranking"]
    d["share_pairs_same_order"] = [
        100 * tau[p]["overall"]["share_same_order"] for p in ("last", "mean")
    ]
    d["temperature_own_range"] = [
        min(generation["temperature_test"]["choice"]["own_temperature"].values()),
        max(generation["temperature_test"]["choice"]["own_temperature"].values()),
    ]
    d["glove_trainable_first_check"] = run_checks("lstm-glove-trainable-s4213-full")[1][0]
    vocab = load(RESULTS / "eda/vocabulary.json")["by_min_count"]["built_on_a_tested_on_b"]
    size = {r["min_count"]: r["vocabulary_size"] for r in vocab}
    unk = {r["min_count"]: 100 * r["held_out_unk_share"] for r in vocab}
    d["vocabulary_more_words_than_min_count_5"] = {k: size[k] - size[5] for k in (3, 2, 1)}
    d["vocabulary_less_unk_than_min_count_5"] = {k: unk[5] - unk[k] for k in (3, 2, 1)}
    d["vocabulary_step"] = {
        f"{a}->{b}": {"more_words": size[b] - size[a], "less_unk": unk[a] - unk[b]}
        for a, b in ((10, 5), (5, 3), (3, 2), (2, 1))
    }
    slices = load(RESULTS / "data/pile_sizes.json")["slices"]

    def balanced(low: float, high: float) -> int:
        part = [x for x in slices if low - 1e-9 <= x["keys"][0] and x["keys"][1] <= high + 1e-9]
        return 2 * min(sum(x["negative"] for x in part), sum(x["positive"] for x in part))

    d["part3_train_if_lm_train_larger"] = {
        "0.7-0.9": round(balanced(0.7, 0.9), -2),
        "0.9-1.0": round(balanced(0.9, 1.0), -2),
    }
    return d


def build_results(out_dir: Path) -> list[str]:
    """Write every results figure and table; return a list of notes for the console."""
    notes = []
    figures, tables = out_dir / "figures", out_dir / "tables"
    piles = load(RESULTS / "data/piles.json")["piles"]
    epoch = piles["lm_train"]["targets"] / 1e6
    facts: dict[str, Any] = {"epoch_targets": piles["lm_train"]["targets"]}
    figure_curves(figures / "curves.pdf", epoch)
    figure_curves_appendix(figures / "curves_appendix.pdf", epoch)
    figure_positions(figures / "positions.pdf")
    figure_part3(figures / "part3.pdf")
    figure_part3_length(figures / "part3_length.pdf")
    table_part1(tables / "part1.tex", facts)
    table_tuning(tables / "tuning.tex")
    table_positions(tables / "positions.tex", facts)
    table_generation(tables / "generation.tex", facts)
    table_samples_all(tables / "samples_all.tex")
    if not table_samples(tables / "samples.tex"):
        notes.append(
            "results/eval/examples_generation.txt not found: kept tables/samples.tex as it is"
        )
    table_part2(tables / "part2.tex", facts)
    table_part2_words(tables / "part2_words.tex", facts)
    table_part3(tables / "part3.tex", facts)
    table_part3_all(tables / "part3_all.tex")
    table_comparisons(tables / "comparisons.tex", facts)
    table_errors(tables / "errors.tex", facts)
    if not exploratory_part3(tables / "exploratory.tex", facts):
        notes.append("data/part3/predictions.npz not found: skipped the exploratory McNemar tests")
    facts["derived"] = derived_facts(facts)
    (out_dir / "facts.json").write_text(
        json.dumps(facts, indent=1, default=float) + "\n", encoding="utf-8"
    )
    return notes
