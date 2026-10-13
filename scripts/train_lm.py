"""Train the neural language models under the locked rules (D0.5, D0.6).

Usage:
    uv run python scripts/train_lm.py --config configs/part1.yaml --models lstm-S --tune
    uv run python scripts/train_lm.py --config configs/part1.yaml --models lstm-S --lr 0.001
    uv run python scripts/train_lm.py --config configs/part2.yaml --models lstm-glove-frozen-s4213

With --tune, each model tries the same three peak learning rates and keeps
stepping half a decade further while its best rate is the smallest or largest
tried (src/a1/training/tuning.py), then keeps the best by validation J. With
full_length_tries: true in the config (locked 12 Oct), every try is a complete
run of `epochs` epochs and the winning try
becomes the model's final run, <model>-full, without being trained again.
Otherwise the tries are shorter runs of `tuning_epochs` and the winner gets a
fresh full run. With --lr only the full run is trained, at that rate; with
neither, each model trains once at the peak_lr of its config entry. Learning
rates are picked on the validation pile; the test pile is never read here.

Part II (configs/part2.yaml). A model entry can also give:

- embedding: a .npy table from scripts/build_embeddings.py, copied into E after
  the model is built (without it, E keeps its random start, as in Part I);
- freeze: true to keep the whole of E fixed during training;
- seed: the seed for this model's starting weights and batch order (default:
  the config's seed). The training sample scored at each check always comes
  from the config's seed, so every run is measured on the same reviews.

Every run records E's fingerprint before and after training and how far it
moved; a run whose fixed table changed stops with an error.

At every validation check, a fixed sample of training reviews as large as the
validation pile is scored the same way (train_sample: true, the default), so
the gap between training and validation J shows overfitting (dropout rule,
12 Oct).

Every run writes results/runs/<run>/run.json (settings, parameter counts,
history, summary, git commit of the code) and adds a row to results/runs.csv.
Best weights go to checkpoints/<run>.pt (git-ignored). A run that already has
a run.json is skipped unless --force is given, so an interrupted batch of runs
can simply be restarted.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # before torch is imported

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import torch  # noqa: E402
import yaml  # noqa: E402

from a1.data.sequences import (  # noqa: E402
    EncodedReviews,
    encode_piles,
    load_piles,
    training_sample,
)
from a1.data.vocab import UNK_ID, Vocabulary  # noqa: E402
from a1.evaluation.timing import pick_device, release_memory  # noqa: E402
from a1.models.build import build_model, count_parameters, describe  # noqa: E402
from a1.models.pretrained import (  # noqa: E402
    embedding_change,
    file_sha256,
    install_table,
    load_table,
    table_sha256,
)
from a1.provenance import code_version, machine_info  # noqa: E402
from a1.training.trainer import TrainConfig, train  # noqa: E402
from a1.training.tuning import (  # noqa: E402
    LEARNING_RATES,
    best_learning_rate,
    next_learning_rate,
)

# Keys of a model entry that are not architecture settings.
RUN_KEYS = ("device", "peak_lr")

RUNS = Path("results/runs")
REGISTRY = Path("results/runs.csv")
CHECKPOINTS = Path("checkpoints")
REGISTRY_FIELDS = [
    "run",
    "model",
    "label",
    "stage",
    "promoted_from",
    "peak_lr",
    "device",
    "batch_tokens",
    "planned_epochs",
    "epochs_done",
    "steps",
    "best_valid_J",
    "best_valid_PPL",
    "seconds",
    "train_seconds",
    "seconds_to_best",
    "train_seconds_to_best",
    "stopped_early",
    "diverged",
    "core_parameters",
    "total_parameters",
    "commit",
    "finished",
]


def log(message: str) -> None:
    print(message, flush=True)


def registry_row(record: dict[str, Any]) -> dict[str, Any]:
    """One results/runs.csv row from a run.json record."""
    summary, config = record["summary"], record["summary"]["config"]
    best = summary["best_valid_J"]
    return {
        "run": record["run"],
        "model": record["model"],
        "label": describe(record["spec"]),
        "stage": record["stage"],
        "promoted_from": record.get("promoted_from", ""),
        "peak_lr": config["peak_lr"],
        "device": record["device"],
        "batch_tokens": config["batch_tokens"],
        "planned_epochs": config["epochs"],
        "epochs_done": round(summary["epochs_done"], 3),
        "steps": summary["steps"],
        "best_valid_J": best,
        "best_valid_PPL": None if best is None else round(math.exp(best), 3),
        "seconds": round(summary["seconds"], 1),
        "train_seconds": round(summary["train_seconds"], 1),
        "seconds_to_best": round(summary["seconds_to_best"], 1),
        "train_seconds_to_best": round(summary["train_seconds_to_best"], 1),
        "stopped_early": summary["stopped_early"],
        "diverged": summary["diverged"],
        "core_parameters": record["parameters"]["core"],
        "total_parameters": record["parameters"]["total"],
        "commit": (record.get("code") or {}).get("commit"),
        "finished": record["finished"],
    }


def save_record(record: dict[str, Any]) -> None:
    """Write results/runs/<run>/run.json and add the run to results/runs.csv."""
    folder = RUNS / record["run"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    new_file = not REGISTRY.exists()
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(registry_row(record))


def previous_best(name: str) -> float | None:
    """The best validation J of a finished run, or None if it has not run yet."""
    path = RUNS / name / "run.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    best = record["summary"]["best_valid_J"]
    return math.inf if best is None or record["summary"]["diverged"] else float(best)


def settings_differ(
    record: dict[str, Any],
    spec: dict[str, Any],
    peak_lr: float,
    epochs: float,
    seed: int,
    table_sha256: str | None,
) -> list[str]:
    """How a finished run differs from the run now asked for (empty if it matches).

    A finished run is only skipped when it is the same run: same model entry,
    peak learning rate, length, seed and input table file.
    """
    config = record["summary"]["config"]
    problems = []
    if not math.isclose(float(config["peak_lr"]), peak_lr, rel_tol=1e-9):
        problems.append(f"peak learning rate {config['peak_lr']} (now {peak_lr})")
    if not math.isclose(float(config["epochs"]), float(epochs), rel_tol=1e-9):
        problems.append(f"epochs {config['epochs']} (now {epochs})")
    if int(config["seed"]) != seed:
        problems.append(f"seed {config['seed']} (now {seed})")
    if record["spec"] != spec:
        problems.append(f"model entry {record['spec']} (now {spec})")
    stored = (record.get("embedding") or {}).get("file_sha256")
    if stored is not None and stored != table_sha256:
        problems.append("the input table file changed since that run")
    return problems


def run_once(
    name: str,
    model_name: str,
    spec: dict[str, Any],
    peak_lr: float,
    epochs: float,
    stage: str,
    data: dict[str, EncodedReviews],
    sample: EncodedReviews | None,
    vocab_size: int,
    settings: dict[str, Any],
    device: torch.device,
    save_weights: bool,
    force: bool,
) -> float:
    """Train one run; return its best validation J (inf if it diverged).

    A run that already finished with the same settings is skipped; one that
    finished with different settings stops the script, so a restart with other
    options never mixes runs (use --force to train it again).
    """
    seed = int(spec.get("seed", settings["seed"]))
    table_path = Path(spec["embedding"]) if spec.get("embedding") else None
    table_file_sha256 = None if table_path is None else file_sha256(table_path)
    if not force and (done := previous_best(name)) is not None:
        record = json.loads((RUNS / name / "run.json").read_text())
        problems = settings_differ(record, spec, peak_lr, epochs, seed, table_file_sha256)
        if problems:
            raise SystemExit(
                f"{name} already exists with other settings: {'; '.join(problems)}. "
                "Use --force to train it again, or keep the finished run."
            )
        log(f"  {name}: already done (best validation J {done}); skipping")
        return done

    torch.manual_seed(seed)
    model = build_model(spec, vocab_size)
    frozen = bool(spec.get("freeze", False))
    table = (
        None
        if table_path is None
        else load_table(table_path, vocab_size, model.embedding.weight.shape[1])
    )
    install_table(model, table, frozen)
    start_table = model.embedding.weight.detach().cpu().clone()
    counts = count_parameters(model)
    config = TrainConfig(
        peak_lr=peak_lr,
        batch_tokens=int(settings["batch_tokens"]),
        epochs=float(epochs),
        seed=seed,
    )
    checkpoint = CHECKPOINTS / f"{name}.pt" if save_weights else None
    log(f"  {name}: {describe(spec)}, peak learning rate {peak_lr}, {epochs} epochs on {device}")
    summary = train(
        model,
        data["lm_train"],
        data["lm_valid"],
        config,
        device,
        checkpoint=checkpoint,
        log=log,
        train_sample=sample,
    )
    end_table = model.embedding.weight.detach().cpu()
    change = embedding_change(start_table, end_table)
    if frozen and not change["unchanged"]:
        raise RuntimeError(f"{name}: the fixed input table changed during training")
    record = {
        "run": name,
        "model": model_name,
        "spec": spec,
        "stage": stage,
        "seed": seed,
        "device": str(device),
        "parameters": counts,
        "embedding": {
            "table": "random start" if table_path is None else str(table_path),
            "file_sha256": table_file_sha256,
            "frozen": frozen,
            "start_sha256": table_sha256(start_table),
            "end_sha256": table_sha256(end_table),
            **change,
        },
        "train_sample_targets": None if sample is None else sample.n_targets,
        "code": code_version(),
        "machine": machine_info(str(device)),
        "summary": summary,
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    save_record(record)
    del model
    release_memory(device)
    best = summary["best_valid_J"]
    log(f"  {name}: best validation J {best} after {summary['seconds'] / 60:.1f} min")
    return math.inf if best is None or summary["diverged"] else float(best)


def promote(winner: str, name: str, force: bool) -> None:
    """Make the winning full-length try the model's final run, without training it again.

    If the final run already comes from this winner, nothing happens. If a later
    try has beaten it (the search was extended), the new winner replaces it. A
    final run that was trained directly (with --lr or a config peak_lr) is never
    replaced without --force.
    """
    path = RUNS / name / "run.json"
    if not force and path.exists():
        current = json.loads(path.read_text()).get("promoted_from")
        if current == winner:
            log(f"  {name}: already done; skipping")
            return
        if current is None:
            raise SystemExit(
                f"{name} was trained directly, not chosen from tries; use --force to replace "
                f"it with {winner}."
            )
        log(f"  {name}: the best try is now {winner} (was {current}); replacing it")
    record = json.loads((RUNS / winner / "run.json").read_text())
    shutil.copyfile(CHECKPOINTS / f"{winner}.pt", CHECKPOINTS / f"{name}.pt")
    record.update(run=name, stage="full", promoted_from=winner)
    save_record(record)
    log(f"  {name}: the try {winner} is the final run (weights in {CHECKPOINTS / name}.pt)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--models", required=True, help="comma-separated names from the config")
    parser.add_argument(
        "--tune", action="store_true", help="learning-rate tries, then the final run"
    )
    parser.add_argument("--lr", type=float, help="skip the tries; full run at this peak rate")
    parser.add_argument("--force", action="store_true", help="redo runs that already exist")
    args = parser.parse_args()
    if args.tune and args.lr is not None:
        parser.error("give --tune or --lr, not both")

    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    full_length_tries = bool(settings.get("full_length_tries", False))
    start = tuple(float(rate) for rate in settings.get("tuning_start", LEARNING_RATES))
    vocab_path = Path(settings["vocabulary"])
    if not vocab_path.exists():
        print(f"Missing {vocab_path}: run scripts/build_vocab.py first", file=sys.stderr)
        return 1
    names = args.models.split(",")
    unknown = [m for m in names if m not in settings["models"]]
    if unknown:
        parser.error(f"not in {args.config}: {', '.join(unknown)}")
    if not args.tune and args.lr is None:
        without = [m for m in names if "peak_lr" not in settings["models"][m]]
        if without:
            parser.error(f"give --tune or --lr, or a peak_lr in the config: {', '.join(without)}")
    tables = [
        Path(settings["models"][m]["embedding"])
        for m in names
        if settings["models"][m].get("embedding")
    ]
    missing = [str(path) for path in tables if not path.exists()]
    if missing:
        print(f"Missing {', '.join(missing)}: run build_embeddings.py first", file=sys.stderr)
        return 1
    vocab = Vocabulary.load(vocab_path)
    piles = load_piles(Path(settings["piles"]))
    data = encode_piles(
        Path(settings["corpus"]), {k: piles[k] for k in ("lm_train", "lm_valid")}, vocab
    )
    sample = (
        training_sample(data["lm_train"], data["lm_valid"].n_targets, int(settings["seed"]))
        if settings.get("train_sample", True)
        else None
    )
    log(
        f"|V| = {len(vocab):,}; training {data['lm_train'].n_targets:,} targets, validation "
        f"{data['lm_valid'].n_targets:,} (<unk> "
        f"{float((data['lm_valid'].ids == UNK_ID).sum()) / data['lm_valid'].n_targets:.2%})"
        + ("" if sample is None else f"; training sample {sample.n_targets:,} targets")
    )
    try_epochs = settings["epochs"] if full_length_tries else settings["tuning_epochs"]

    for model_name in names:
        entry = settings["models"][model_name]
        device = pick_device(entry.get("device", settings.get("device")))
        spec = {k: v for k, v in entry.items() if k not in RUN_KEYS}
        log(f"\n{model_name}: {describe(spec)} on {device}")
        final = f"{model_name}-full"
        if args.tune:
            tries: dict[float, float] = {}
            while (rate := next_learning_rate(tries, start)) is not None:
                tries[rate] = run_once(
                    f"{model_name}-try-lr{rate:g}",
                    model_name,
                    spec,
                    rate,
                    try_epochs,
                    "try",
                    data,
                    sample,
                    len(vocab),
                    settings,
                    device,
                    save_weights=full_length_tries,
                    force=args.force,
                )
            peak_lr = best_learning_rate(tries)
            (RUNS / f"{model_name}-tuning.json").write_text(
                json.dumps(
                    {
                        "tries": {str(k): v for k, v in tries.items()},
                        "chosen": peak_lr,
                        "start": list(start),
                        "full_length_tries": full_length_tries,
                        "try_epochs": try_epochs,
                    },
                    indent=2,
                )
                + "\n"
            )
            log(f"  {model_name}: chose peak learning rate {peak_lr} from {sorted(tries)}")
            if full_length_tries:
                promote(f"{model_name}-try-lr{peak_lr:g}", final, args.force)
                continue
        else:
            peak_lr = args.lr if args.lr is not None else float(entry["peak_lr"])
        run_once(
            final,
            model_name,
            spec,
            peak_lr,
            settings["epochs"],
            "full",
            data,
            sample,
            len(vocab),
            settings,
            device,
            save_weights=True,
            force=args.force,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
