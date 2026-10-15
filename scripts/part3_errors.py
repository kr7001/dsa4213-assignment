"""Part III error analysis (D3.4): measured groups and seeded samples of mistakes.

Usage:
    uv run python scripts/part3_errors.py 2>&1 | tee results/logs/part3_errors.log

Run it after scripts/part3.py. It reads configs/part3_errors.yaml, the Part III
settings in configs/part3.yaml, and the per-review test predictions of the
Part III run (data/part3/predictions.npz). It rebuilds the balanced test
reviews and stops unless they are exactly the reviews the predictions belong
to. Nothing here changes a classifier or chooses anything. It writes:

- results/part3/errors.json (numbers only): accuracy inside every measured
  group for the main representation, the reference and the others listed;
  how often the main representation and the reference are right or wrong
  together; the ranking agreement between test J and accuracy inside the
  hard groups; and the review ids of every sampled review.
- results/part3/examples_errors.txt: the sampled reviews with their text, for
  reading and categorising (git-ignored, like every examples file).
- results/part3/error_categories.csv: one row per sampled mistake with an
  empty category column to fill in (no text).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from a1.data.sequences import load_piles
from a1.data.vocab import Vocabulary
from a1.part3.data import balance, read_part3
from a1.part3.errors import (
    CATEGORIES,
    CONTRAST,
    NEGATION,
    describe_review,
    disagreement,
    draw,
    group_accuracy,
    groups,
    star_group,
)
from a1.part3.stats import pair_agreement
from a1.provenance import code_version, machine_info


def log(message: str) -> None:
    print(message, flush=True)


def texts_of(corpus: Path, review_ids: set[int]) -> dict[int, str]:
    """The text of each requested review (as the models read it: cleaned and tokenized)."""
    found: dict[int, str] = {}
    with gzip.open(corpus, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["id"] in review_ids:
                found[int(row["id"])] = row["text"]
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/part3_errors.yaml"))
    args = parser.parse_args(argv)
    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = yaml.safe_load(Path(settings["part3"]).read_text(encoding="utf-8"))
    seed = int(settings["seed"])
    vocab = Vocabulary.load(Path(config["vocabulary"]))
    corpus = Path(config["corpus"])

    piles = load_piles(Path(config["piles"]))
    part3 = {name: piles[name] for name in ("part3_train", "part3_test")}
    others = {name: pile for name, pile in piles.items() if name not in part3}
    reading = read_part3(corpus, part3, others, vocab)
    train = balance(reading.sets["part3_train"], int(config["seed"]))
    test = balance(reading.sets["part3_test"], int(config["seed"]))

    with np.load(Path(config["cache"]) / "predictions.npz", allow_pickle=False) as stored:
        if not np.array_equal(stored["review_ids"], test.reviews.review_ids) or not np.array_equal(
            stored["labels"], test.labels
        ):
            raise SystemExit("The predictions belong to other test reviews; rerun scripts/part3.py")
        names = stored["names"].tolist()
        predicted = {name: stored["predicted"][i] for i, name in enumerate(names)}
        probability = {name: stored["probability"][i] for i, name in enumerate(names)}
    correct = {name: predicted[name] == test.labels for name in names}
    main_name, reference = settings["main"], settings["reference"]
    reported = [main_name, reference, *settings["also"]]
    missing = [name for name in reported if name not in correct]
    if missing:
        raise SystemExit(f"No predictions for {missing}")

    log("== Measured groups (D3.4, part 1) ==")
    length_groups = [tuple(group) for group in config["length_groups"]]
    masks = groups(
        test,
        vocab,
        set(train.text_hash.tolist()),
        length_groups,
        float(settings["heavy_unk_share"]),
    )
    accuracy = {name: group_accuracy(correct[name], masks) for name in reported}
    width = max(len(name) for name in masks)
    log(f"  {'group':{width}s} {'reviews':>8s}  " + "  ".join(f"{n[:22]:>22s}" for n in reported))
    for group in masks:
        values = "  ".join(
            f"{'-':>22s}"
            if accuracy[n][group]["accuracy"] is None
            else f"{accuracy[n][group]['accuracy']:22.4f}"
            for n in reported
        )
        log(f"  {group:{width}s} {accuracy[main_name][group]['reviews']:8d}  {values}")

    together = disagreement(correct[main_name], correct[reference])
    log(f"\n{main_name} against {reference}: {together}")

    test_scores = json.loads(Path(config["test_scores"]).read_text())
    test_j = np.asarray([test_scores["models"][m]["test"]["J"] for m in config["ranking"]])
    ranking: dict[str, dict[str, Any]] = {}
    for pooling in config["poolings"]:
        reps = [f"{m}/{config['layer']}/{pooling}" for m in config["ranking"]]
        ranking[pooling] = {}
        for group in ["all", *settings["ranking_groups"]]:
            mask = masks[group]
            if not mask.any():
                ranking[pooling][group] = None
                continue
            scores = np.asarray([correct[rep][mask].mean() for rep in reps])
            agreement = pair_agreement(test_j, scores)
            agreement["reviews"] = int(mask.sum())
            agreement["accuracy"] = dict(zip(config["ranking"], scores.tolist(), strict=True))
            ranking[pooling][group] = agreement
            log(
                f"  ranking, {pooling:4s}, {group:9s}: {agreement['same_order']} of "
                f"{agreement['pairs']} pairs in the same order ({int(mask.sum()):,} reviews)"
            )

    log("\n== Samples for reading (D3.4, part 2) ==")
    rng = np.random.default_rng(seed)
    main_right, reference_right = correct[main_name], correct[reference]
    samples = {
        "mistakes": draw(np.flatnonzero(~main_right), int(settings["samples"]["mistakes"]), rng),
        "main_right_reference_wrong": draw(
            np.flatnonzero(main_right & ~reference_right),
            int(settings["samples"]["disagreements"]),
            rng,
        ),
        "reference_right_main_wrong": draw(
            np.flatnonzero(~main_right & reference_right),
            int(settings["samples"]["disagreements"]),
            rng,
        ),
    }
    shown = [main_name, reference]
    described = {
        kind: [
            describe_review(test, int(i), masks, {n: probability[n][i] for n in shown})
            for i in positions
        ]
        for kind, positions in samples.items()
    }
    wanted = {row["review_id"] for rows in described.values() for row in rows}
    texts = texts_of(corpus, wanted)

    results = Path(config["results"])
    results.mkdir(parents=True, exist_ok=True)
    lines = [
        "Part III error analysis (D3.4): seeded random samples. Review text: keep this file",
        "out of git and out of the report except for short quoted examples.",
        "",
        f"Main representation: {main_name}; reference: {reference}.",
        "probability_positive is each classifier's probability that the review is positive.",
        "",
        "Categories for the mistakes (one main category each; if several fit, the first wins):",
        *(f"  {k}. {name}" for k, name in enumerate(CATEGORIES, start=1)),
        "",
    ]
    titles = {
        "mistakes": "Mistakes of the main representation",
        "main_right_reference_wrong": f"{main_name} right, {reference} wrong",
        "reference_right_main_wrong": f"{reference} right, {main_name} wrong",
    }
    for kind, rows in described.items():
        lines.append(f"===== {titles[kind]} ({len(rows)}) =====")
        for k, row in enumerate(rows, start=1):
            flags = ", ".join(row["flags"]) or "none"
            probabilities = ", ".join(
                f"{name}: {p:.3f}" for name, p in row["probability_positive"].items()
            )
            lines += [
                f"[{kind} {k}] review {row['review_id']}, {star_group(row['stars'])} "
                f"(label {row['label']}), {row['words']} words; groups: {flags}",
                f"  probability_positive: {probabilities}",
                f"  {texts[row['review_id']]}",
                "",
            ]
    examples = results / "examples_errors.txt"
    examples.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"Wrote {examples} (review text; git-ignored)")

    sheet = results / "error_categories.csv"
    with sheet.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample", "review_id", "stars", "label", "category", "reason"])
        for k, row in enumerate(described["mistakes"], start=1):
            writer.writerow([k, row["review_id"], row["stars"], row["label"], "", ""])
    log(f"Wrote {sheet} (one row per sampled mistake, category to fill in; no text)")

    payload = {
        "main": main_name,
        "reference": reference,
        "group_definitions": {
            "negation_words": sorted(NEGATION),
            "negation_rule": "a negation word, or a token ending in n't",
            "contrast_words": sorted(CONTRAST),
            "heavy_unk_share": float(settings["heavy_unk_share"]),
            "repeat": "the exact text of a part3_train review",
        },
        "group_accuracy": accuracy,
        "main_against_reference": together,
        "ranking_in_groups": ranking,
        "categories": list(CATEGORIES),
        "samples": described,
        "seed": seed,
        "code": code_version(),
        "machine": machine_info(None),
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    path = results / "errors.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    log(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
