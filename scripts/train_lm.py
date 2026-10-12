"""Train the neural language models under the locked rules (D0.5, D0.6).

Usage:
    uv run python scripts/train_lm.py --config configs/part1.yaml --models rnn,lstm --tune
    uv run python scripts/train_lm.py --config configs/part1.yaml --models lstm --lr 0.001

With --tune, each model first tries the same three peak learning rates in short
runs (configs: tuning_epochs), plus one more step if the best is at an edge
(src/a1/training/tuning.py), then trains the full run (configs: epochs) with
the winner. With --lr it only trains the full run at that rate. Learning rates
are picked on the validation pile; the test pile is never read here.

Every run writes results/runs/<run>/run.json (settings, parameter counts,
history, summary) and adds a row to results/runs.csv. The best weights of a
full run go to checkpoints/<run>.pt (git-ignored). A run that already has a
run.json is skipped unless --force is given, so an interrupted batch of runs
can simply be restarted.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # before torch is imported

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import torch  # noqa: E402
import yaml  # noqa: E402

from a1.data.sequences import EncodedReviews, encode_piles, load_piles  # noqa: E402
from a1.data.vocab import UNK_ID, Vocabulary  # noqa: E402
from a1.evaluation.timing import pick_device, release_memory  # noqa: E402
from a1.models.build import build_model, count_parameters, describe  # noqa: E402
from a1.training.trainer import TrainConfig, train  # noqa: E402
from a1.training.tuning import best_learning_rate, next_learning_rate  # noqa: E402

RUNS = Path("results/runs")
REGISTRY = Path("results/runs.csv")
CHECKPOINTS = Path("checkpoints")
REGISTRY_FIELDS = [
    "run",
    "model",
    "label",
    "stage",
    "peak_lr",
    "device",
    "batch_tokens",
    "planned_epochs",
    "epochs_done",
    "steps",
    "best_valid_J",
    "best_valid_PPL",
    "seconds",
    "seconds_to_best",
    "stopped_early",
    "diverged",
    "core_parameters",
    "total_parameters",
    "finished",
]


def log(message: str) -> None:
    print(message, flush=True)


def run_once(
    name: str,
    model_name: str,
    spec: dict[str, Any],
    peak_lr: float,
    epochs: float,
    stage: str,
    data: dict[str, EncodedReviews],
    vocab_size: int,
    settings: dict[str, Any],
    device: torch.device,
    force: bool,
) -> float:
    """Train one run; return its best validation J (inf if it diverged)."""
    folder = RUNS / name
    if (folder / "run.json").exists() and not force:
        previous = json.loads((folder / "run.json").read_text())
        best = previous["summary"]["best_valid_J"]
        log(f"  {name}: already done (best validation J {best}); skipping")
        return math.inf if best is None else float(best)

    torch.manual_seed(int(settings["seed"]))
    model = build_model(spec, vocab_size)
    counts = count_parameters(model)
    config = TrainConfig(
        peak_lr=peak_lr,
        batch_tokens=int(settings["batch_tokens"]),
        epochs=float(epochs),
        seed=int(settings["seed"]),
    )
    checkpoint = CHECKPOINTS / f"{name}.pt" if stage == "full" else None
    log(f"  {name}: {describe(spec)}, peak learning rate {peak_lr}, {epochs} epochs on {device}")
    summary = train(
        model, data["lm_train"], data["lm_valid"], config, device, checkpoint=checkpoint, log=log
    )
    best = summary["best_valid_J"]
    finished = datetime.now().isoformat(timespec="seconds")
    folder.mkdir(parents=True, exist_ok=True)
    record = {
        "run": name,
        "model": model_name,
        "spec": spec,
        "stage": stage,
        "device": str(device),
        "parameters": counts,
        "summary": summary,
        "finished": finished,
    }
    (folder / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    row = {
        "run": name,
        "model": model_name,
        "label": describe(spec),
        "stage": stage,
        "peak_lr": peak_lr,
        "device": str(device),
        "batch_tokens": config.batch_tokens,
        "planned_epochs": epochs,
        "epochs_done": round(summary["epochs_done"], 3),
        "steps": summary["steps"],
        "best_valid_J": best,
        "best_valid_PPL": None if best is None else round(math.exp(best), 3),
        "seconds": round(summary["seconds"], 1),
        "seconds_to_best": round(summary["seconds_to_best"], 1),
        "stopped_early": summary["stopped_early"],
        "diverged": summary["diverged"],
        "core_parameters": counts["core"],
        "total_parameters": counts["total"],
        "finished": finished,
    }
    new_file = not REGISTRY.exists()
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    del model
    release_memory(device)
    log(f"  {name}: best validation J {best} after {summary['seconds'] / 60:.1f} min")
    return math.inf if best is None or summary["diverged"] else float(best)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--models", required=True, help="comma-separated names from the config")
    parser.add_argument(
        "--tune", action="store_true", help="learning-rate tries, then the full run"
    )
    parser.add_argument("--lr", type=float, help="skip the tries; full run at this peak rate")
    parser.add_argument("--force", action="store_true", help="redo runs that already exist")
    args = parser.parse_args()
    if not args.tune and args.lr is None:
        parser.error("give --tune or --lr")

    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    vocab_path = Path(settings["vocabulary"])
    if not vocab_path.exists():
        print(f"Missing {vocab_path}: run scripts/build_vocab.py first", file=sys.stderr)
        return 1
    vocab = Vocabulary.load(vocab_path)
    piles = load_piles(Path(settings["piles"]))
    data = encode_piles(
        Path(settings["corpus"]), {k: piles[k] for k in ("lm_train", "lm_valid")}, vocab
    )
    log(
        f"|V| = {len(vocab):,}; training {data['lm_train'].n_targets:,} targets, validation "
        f"{data['lm_valid'].n_targets:,} (<unk> "
        f"{float((data['lm_valid'].ids == UNK_ID).sum()) / data['lm_valid'].n_targets:.2%})"
    )

    for model_name in args.models.split(","):
        spec = settings["models"][model_name]
        device = pick_device(spec.get("device", settings.get("device")))
        spec = {k: v for k, v in spec.items() if k != "device"}
        log(f"\n{model_name}: {describe(spec)} on {device}")
        if args.tune:
            tries: dict[float, float] = {}
            while (rate := next_learning_rate(tries)) is not None:
                tries[rate] = run_once(
                    f"{model_name}-try-lr{rate:g}",
                    model_name,
                    spec,
                    rate,
                    settings["tuning_epochs"],
                    "try",
                    data,
                    len(vocab),
                    settings,
                    device,
                    args.force,
                )
            peak_lr = best_learning_rate(tries)
            (RUNS / f"{model_name}-tuning.json").write_text(
                json.dumps(
                    {"tries": {str(k): v for k, v in tries.items()}, "chosen": peak_lr}, indent=2
                )
                + "\n"
            )
            log(f"  {model_name}: chose peak learning rate {peak_lr} from {sorted(tries)}")
        else:
            peak_lr = args.lr
        run_once(
            f"{model_name}-full",
            model_name,
            spec,
            peak_lr,
            settings["epochs"],
            "full",
            data,
            len(vocab),
            settings,
            device,
            args.force,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
