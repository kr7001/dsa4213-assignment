"""The test run (pilot): speed, memory and gradients before the sizes are fixed.

Usage:
    uv run python scripts/pilot.py [--config configs/pilot.yaml] [--models rnn-S,lstm-S]
                                   [--device mps] [--quick]

It only reads products with a split key below 0.5, all inside the future
training pile: pilot training reviews (key below 0.45) and pilot validation
reviews (0.45 to 0.5). No validation or test pile is touched.

For each model in the config it:
1. counts parameters: total, word tables and core;
2. times full training steps at each batch size (targets per second, memory);
3. times two batch sizes on the CPU too, since small recurrent models can run
   faster there, and the cost comparison needs the same hardware for every model;
4. trains for a few hundred steps with the locked rules, keeping every unclipped
   gradient norm, and measures the gradient noise scale (McCandlish et al., 2018)
   three times along the way;
5. times one pass over the pilot validation reviews.

It writes results/pilot/pilot.json (numbers only, no review text) and prints a
summary. --quick runs a tiny version to check that everything works.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # before torch is imported

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from a1.data.sequences import PILOT_TRAIN, PILOT_VALID, count_tokens, encode_pile  # noqa: E402
from a1.data.vocab import UNK_ID, Vocabulary  # noqa: E402
from a1.evaluation.perplexity import evaluate  # noqa: E402
from a1.evaluation.timing import (  # noqa: E402
    is_out_of_memory,
    pick_device,
    release_memory,
    synchronize,
    training_speed,
)
from a1.models.build import build_model, count_parameters, describe  # noqa: E402
from a1.training.noise import measure_noise_scale  # noqa: E402
from a1.training.trainer import TrainConfig, make_optimizer, train  # noqa: E402

DEFAULT_CONFIG = Path("configs/pilot.yaml")
RESULTS = Path("results/pilot")
QUICK = {  # a tiny version of every setting, to check the script end to end
    "speed": {"batch_tokens": [1024, 2048], "timed_steps": 2, "cpu_batch_tokens": [1024]},
    "short_run": {"batch_tokens": 1024, "steps": 20, "peak_lr": 0.001},
    "noise": {"micro_tokens": 1024, "micro_batches": 4, "after_steps": [5, 10, 20]},
}


def machine_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_threads": torch.get_num_threads(),
    }
    if sys.platform == "darwin":
        for key, name in (("chip", "machdep.cpu.brand_string"), ("memory_bytes", "hw.memsize")):
            with contextlib.suppress(OSError, subprocess.CalledProcessError):
                info[key] = subprocess.run(
                    ["sysctl", "-n", name], capture_output=True, text=True, check=True
                ).stdout.strip()
    return info


def log(message: str) -> None:
    print(message, flush=True)


def speed_table(spec: dict, vocab_size: int, data, settings: dict, device, seed: int) -> list:
    """Training speed at each batch size, smallest first; stops at the first memory error."""
    rows = []
    for tokens in settings["batch_tokens"]:
        torch.manual_seed(seed)
        model = build_model(spec, vocab_size).to(device)
        optimizer = make_optimizer(model, TrainConfig())
        try:
            row = training_speed(
                model,
                optimizer,
                data,
                tokens,
                device,
                steps=settings["timed_steps"],
                seed=seed,
            )
        except RuntimeError as error:
            if not is_out_of_memory(error):
                raise
            row = {"batch_tokens": tokens, "error": "out of memory"}
        finally:
            del model, optimizer
            release_memory(device)
        rows.append(row)
        shown = (
            f"{row['targets_per_second']:,.0f} targets/s, {row['seconds_per_step']:.3f} s/step"
            if "error" not in row
            else row["error"]
        )
        log(f"    batch {tokens:>6,} tokens: {shown}")
        if "error" in row:
            break
    return rows


def run_model(
    spec: dict,
    vocab: Vocabulary,
    train_data,
    valid_data,
    settings: dict,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    name = spec["name"]
    config = {k: v for k, v in spec.items() if k != "name"}
    counts = count_parameters(build_model(config, len(vocab)))
    log(f"\n{name}: {describe(config)}  core {counts['core']:,}  total {counts['total']:,}")
    result: dict[str, Any] = {"name": name, "config": config, "parameters": counts}

    log("  speed on " + str(device))
    result["speed"] = speed_table(config, len(vocab), train_data, settings["speed"], device, seed)
    if device.type != "cpu":  # the same steps on the CPU, for every model
        cpu = dict(settings["speed"], batch_tokens=settings["speed"]["cpu_batch_tokens"])
        log("  speed on cpu")
        result["speed_cpu"] = speed_table(
            config, len(vocab), train_data, cpu, torch.device("cpu"), seed
        )

    run = settings["short_run"]
    noise_settings = settings["noise"]
    log(
        f"  short run: {run['steps']} steps at {run['batch_tokens']:,} tokens, "
        f"peak learning rate {run['peak_lr']}"
    )
    torch.manual_seed(seed)
    model = build_model(config, len(vocab))
    noise: list[dict[str, Any]] = []

    def after_step(step: int, current: torch.nn.Module) -> None:
        if step in noise_settings["after_steps"]:
            estimate = measure_noise_scale(
                current,
                train_data,
                noise_settings["micro_tokens"],
                noise_settings["micro_batches"],
                device,
                seed=seed + step,
            )
            noise.append({"after_step": step} | estimate.as_dict())
            log(f"    noise scale after step {step}: {estimate.b_simple:,.0f} targets")

    train_config = TrainConfig(
        peak_lr=run["peak_lr"],
        batch_tokens=run["batch_tokens"],
        epochs=1.0,
        max_steps=run["steps"],
        log_every=max(1, run["steps"] // 12),
        seed=seed,
    )
    summary = train(model, train_data, None, train_config, device, after_step=after_step, log=log)
    norms = summary["gradient_norms"]
    log(
        f"    gradient norms, all steps: mean {norms['mean']:.3f}, p99 {norms['p99']:.3f}, "
        f"clipped {norms['share_clipped']:.1%}, 1.0 in safe range: "
        f"{norms['threshold_in_safe_range']}"
    )
    log(
        f"    gradient norms, second half: mean {norms['late_mean']:.3f}, "
        f"p99 {norms['late_p99']:.3f}, clipped {norms['late_share_clipped']:.1%}, "
        f"1.0 in safe range: {norms['late_threshold_in_safe_range']}"
    )
    result["short_run"] = {k: v for k, v in summary.items() if k != "config"}
    result["noise_scale"] = noise

    synchronize(device)
    start = time.perf_counter()
    scored = evaluate(model, valid_data, run["batch_tokens"], device)
    synchronize(device)
    seconds = time.perf_counter() - start
    result["validation_pass"] = {
        "seconds": seconds,
        "targets": scored["targets"],
        "targets_per_second": scored["targets"] / seconds,
        "J_after_short_run": scored["J"],
    }
    log(
        f"  validation pass: {seconds:.1f} s for {scored['targets']:,} targets, "
        f"J {scored['J']:.3f} after the short run"
    )
    del model
    release_memory(device)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--models", help="comma-separated names to run (default: all)")
    parser.add_argument("--device", help="mps, cuda or cpu (default: the best available)")
    parser.add_argument("--quick", action="store_true", help="tiny settings, to check the script")
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()

    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.quick:
        settings.update(QUICK)
    corpus = Path(settings["corpus"])
    if not corpus.exists():
        print(f"Missing {corpus}: run scripts/prepare_data.py first", file=sys.stderr)
        return 1
    models = settings["models"]
    if args.models:
        wanted = set(args.models.split(","))
        models = [m for m in models if m["name"] in wanted]
    device = pick_device(args.device)
    seed = int(settings.get("seed", 4213))
    started = time.perf_counter()

    log(f"Device: {device}. Reading the pilot reviews (split key below 0.5)...")
    counts, _ = count_tokens(corpus, PILOT_TRAIN)
    vocab = Vocabulary.build(counts)
    vocab.save(Path(settings["vocabulary"]))
    train_data = encode_pile(corpus, PILOT_TRAIN, vocab)
    valid_data = encode_pile(corpus, PILOT_VALID, vocab)
    if args.quick:
        train_data = train_data.subset(np.arange(min(len(train_data), 3000)))
        valid_data = valid_data.subset(np.arange(min(len(valid_data), 300)))
    data_info = {
        "vocabulary_size": len(vocab),
        "train_reviews": len(train_data),
        "train_targets": train_data.n_targets,
        "valid_reviews": len(valid_data),
        "valid_targets": valid_data.n_targets,
        "valid_unk_share": float((valid_data.ids == UNK_ID).sum()) / valid_data.n_targets,
        "longest_review_tokens": int(train_data.lengths.max()),
    }
    log(
        f"|V| = {len(vocab):,}; pilot training {len(train_data):,} reviews "
        f"({train_data.n_targets:,} targets); validation {len(valid_data):,} reviews"
    )

    results = []
    for spec in models:
        results.append(run_model(spec, vocab, train_data, valid_data, settings, device, seed))
        output = {
            "machine": machine_info(device),
            "data": data_info,
            "settings": {k: v for k, v in settings.items() if k != "models"},
            "quick": args.quick,
            "models": results,
            "minutes": (time.perf_counter() - started) / 60,
        }
        args.results.mkdir(parents=True, exist_ok=True)
        (args.results / "pilot.json").write_text(json.dumps(output, indent=2) + "\n")

    log("\nSummary (targets per second at each batch size; noise scale in targets)")
    for result in results:
        speeds = ", ".join(
            f"{row['batch_tokens'] // 1024}k: "
            + (f"{row['targets_per_second']:,.0f}" if "error" not in row else "out of memory")
            for row in result["speed"]
        )
        noise = ", ".join(f"{row['b_simple']:,.0f}" for row in result["noise_scale"])
        cpu = ", ".join(
            f"{row['batch_tokens'] // 1024}k: "
            + (f"{row['targets_per_second']:,.0f}" if "error" not in row else "out of memory")
            for row in result.get("speed_cpu", [])
        )
        norms = result["short_run"]["gradient_norms"]
        core = result["parameters"]["core"]
        log(f"  {result['name']:<14} core {core:>9,}  {device.type} {speeds}")
        if cpu:
            log(f"  {'':<14} cpu {cpu}")
        log(
            f"  {'':<14} noise scale {noise}; clipped {norms['share_clipped']:.0%} "
            f"(second half {norms['late_share_clipped']:.0%}); "
            f"mean norm, second half {norms['late_mean']:.2f}"
        )
    log(
        f"\nDone in {(time.perf_counter() - started) / 60:.1f} min. "
        f"Wrote {args.results / 'pilot.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
