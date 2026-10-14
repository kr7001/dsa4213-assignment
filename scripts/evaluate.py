"""Final numbers for Parts I and II: test scores, loss by position, costs and generated text.

Usage:
    uv run python scripts/evaluate.py [--stages test,costs,generation]
        [--config configs/eval.yaml] [--generation configs/generation.yaml]

Run it once the settings are locked (D1.5 to D1.7) and the code is committed.
It is the only script that reads the test pile (lm_test), and nothing is
chosen on it. Each stage writes one file to results/eval/:

- test.json (D0.5, D1.5): every final model's test J and PPL, with the target
  count checked to be the same for all. Before that, each checkpoint scores the
  validation pile again and must reproduce the best validation J its run
  recorded, which shows the weights are the ones training kept. Also J by
  target position, grouped 1-4, 5-8, ..., 129-513.
- costs.json (D1.7): training time from the run records (final run, time to the
  best checkpoint, all learning-rate tries), parameters, checkpoint size, FLOPs
  per prediction, and speed and memory on the common workload, scoring all of
  lm_test: neural models on the GPU, the S models also on the CPU, the 5-gram
  on the CPU. One untimed pass, then 5 timed passes; the median is reported.
- generation.json (D1.6): the temperature test on lm_valid, the text scores at
  the chosen T on lm_test openings beside the real continuations, and nine
  samples per prompt for each model, the middle one by its own perplexity
  shown. Written text goes to results/eval/examples_generation.txt, which git
  ignores; the JSON keeps only numbers and the prompts.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # before torch is imported

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch import nn  # noqa: E402

from a1.data.sequences import EncodedReviews, encode_piles, load_piles  # noqa: E402
from a1.data.vocab import BOS_ID, UNK_ID, Vocabulary  # noqa: E402
from a1.evaluation.flops import forward_flops  # noqa: E402
from a1.evaluation.perplexity import evaluate  # noqa: E402
from a1.evaluation.positions import (  # noqa: E402
    group_positions,
    mean_context,
    neural_position_sums,
    ngram_position_sums,
)
from a1.evaluation.timing import (  # noqa: E402
    ngram_scoring_speed,
    pick_device,
    release_memory,
    scoring_speed,
)
from a1.generation import (  # noqa: E402
    bootstrap_scores,
    choose_temperature,
    detokenize,
    distinct_halves,
    generate,
    make_sampler,
    middle_sample,
    text_scores,
)
from a1.models.build import build_model, count_parameters, describe  # noqa: E402
from a1.models.ngram import AddK, KneserNey, NgramCounts  # noqa: E402
from a1.models.pretrained import table_sha256  # noqa: E402
from a1.provenance import code_version, machine_info, peak_process_memory  # noqa: E402

NGRAM = "kneser-ney"
ADD_K = "add-k"
VALID_TOLERANCE = 1e-4  # a reloaded checkpoint may differ from its record by float noise only
REAL_CHUNK = 64  # reviews per forward pass when scoring real continuations


def log(message: str) -> None:
    print(message, flush=True)


@dataclass
class Context:
    config: dict[str, Any]
    vocab: Vocabulary
    data: dict[str, EncodedReviews]
    table: NgramCounts
    kn: KneserNey
    addk: AddK
    ngram_record: dict[str, Any]
    counting_seconds: float
    device: torch.device
    runs: Path
    checkpoints: Path
    results: Path
    checks: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def batch_tokens(self) -> int:
        return int(self.config["batch_tokens"])


def final_models(config: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for group in config["models"].values():
        names.extend(name for name in group if name not in names)
    return names


def load_record(ctx: Context, name: str) -> dict[str, Any]:
    return json.loads((ctx.runs / name / "run.json").read_text())


def load_neural(ctx: Context, name: str, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """A final model with its best weights, checked against its run record."""
    record = load_record(ctx, name)
    model = build_model(record["spec"], len(ctx.vocab))
    state = torch.load(ctx.checkpoints / f"{name}.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    expected = (record.get("embedding") or {}).get("end_sha256")
    if expected and table_sha256(model.embedding.weight.detach().cpu()) != expected:
        raise RuntimeError(
            f"{name}: the checkpoint's input table is not the one its run ended with"
        )
    return model.to(device).eval(), record


def load_model(ctx: Context, name: str) -> nn.Module | KneserNey:
    return ctx.kn if name == NGRAM else load_neural(ctx, name, ctx.device)[0]


def free(model: Any, device: torch.device) -> None:
    del model
    release_memory(device)


def save(ctx: Context, filename: str, payload: dict[str, Any]) -> Path:
    payload = {
        **payload,
        "code": code_version(),
        "machine": machine_info(str(ctx.device)),
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    ctx.results.mkdir(parents=True, exist_ok=True)
    path = ctx.results / filename
    path.write_text(json.dumps(payload, indent=2) + "\n")
    log(f"Wrote {path}")
    return path


def score_row(j: float, targets: int, unk: float) -> dict[str, Any]:
    return {"J": j, "PPL": math.exp(j), "targets": targets, "unk_share": unk}


# ---------------------------------------------------------------------------
# Stage 1: test scores and loss by position (D0.5, D1.5)


def check_models(ctx: Context) -> dict[str, dict[str, float]]:
    """Before any test data is used: every model must be the one training kept.

    Each checkpoint's input table must match its run's fingerprint (Part II), and
    each model must reproduce, on the validation pile, the best validation J its
    run recorded (the n-gram models: the J in ngram.json). Any failure stops the
    script here, so a wrong checkpoint never produces a test number.
    """
    valid = ctx.data["lm_valid"]
    recorded = {(r["method"], r["order"]): r["valid"]["J"] for r in ctx.ngram_record["validation"]}
    checks: dict[str, dict[str, float]] = {}
    problems: list[str] = []
    for name, model, method in ((NGRAM, ctx.kn, "kneser_ney"), (ADD_K, ctx.addk, "add_k")):
        checks[name] = {
            "valid_J": float(-model.log_probs(valid).mean()),
            "valid_J_recorded": float(recorded[(method, model.order)]),
        }
    for name in final_models(ctx.config):
        try:
            model, record = load_neural(ctx, name, ctx.device)
        except (RuntimeError, FileNotFoundError, KeyError) as error:
            problems.append(f"{name}: {error}")
            continue
        checks[name] = {
            "valid_J": evaluate(model, valid, ctx.batch_tokens, ctx.device)["J"],
            "valid_J_recorded": float(record["summary"]["best_valid_J"]),
        }
        free(model, ctx.device)
    for name, check in checks.items():
        check["difference"] = check["valid_J"] - check["valid_J_recorded"]
        log(
            f"  {name}: validation J {check['valid_J']:.5f} "
            f"(recorded {check['valid_J_recorded']:.5f})"
        )
        if abs(check["difference"]) > VALID_TOLERANCE:
            problems.append(
                f"{name}: validation J differs from its record by {check['difference']:.6f}"
            )
    if problems:
        raise SystemExit("Stopped before using the test pile:\n  " + "\n  ".join(problems))
    log(f"  all {len(checks)} models reproduce their validation J (within {VALID_TOLERANCE})")
    return checks


def stage_test(ctx: Context) -> dict[str, Any]:
    test = ctx.data["lm_test"]
    unk = float((test.ids == UNK_ID).sum()) / test.n_targets
    rows: dict[str, dict[str, Any]] = {}
    for name, model in ((NGRAM, ctx.kn), (ADD_K, ctx.addk)):
        log_probs = model.log_probs(test)
        if log_probs.size != test.n_targets:
            raise AssertionError(
                f"{name} scored {log_probs.size} targets, expected {test.n_targets}"
            )
        sums, counts = ngram_position_sums(log_probs, test)
        rows[name] = {
            "family": "ngram",
            "label": f"{name} {model.order}-gram",
            **ctx.checks[name],
            "test": score_row(float(-log_probs.mean()), int(log_probs.size), unk),
            "by_position": group_positions(sums, counts),
        }
        log(f"  {name}: test J {rows[name]['test']['J']:.4f}")

    for name in final_models(ctx.config):
        model, record = load_neural(ctx, name, ctx.device)
        result = evaluate(model, test, ctx.batch_tokens, ctx.device, by_position=True)
        sums, counts = neural_position_sums(result)
        rows[name] = {
            "family": record["spec"]["family"],
            "label": describe(record["spec"]),
            **ctx.checks[name],
            "test": score_row(result["J"], result["targets"], result["unk_share"]),
            "by_position": group_positions(sums, counts),
            "J_by_position": [None if math.isnan(x) else x for x in result["J_by_position"]],
            "targets_by_position": result["targets_by_position"],
        }
        log(f"  {name}: test J {result['J']:.4f}")
        free(model, ctx.device)

    counts = {row["test"]["targets"] for row in rows.values()}
    if counts != {test.n_targets}:
        raise AssertionError(f"Models scored different numbers of targets: {sorted(counts)}")
    output = {
        "pile": "lm_test",
        "reviews": len(test),
        "targets": test.n_targets,
        "unk_share": unk,
        "checkpoints_reproduce_valid_J": True,  # check_models stops the script otherwise
        "valid_J_tolerance": VALID_TOLERANCE,
        "models": rows,
    }
    save(ctx, "test.json", output)
    log("\nTest J (PPL) | validation J recomputed | J by position, groups 1-4 ... 129-513")
    for name, row in rows.items():
        groups = " ".join(
            f"{g['J']:.3f}" if g["J"] is not None else "-" for g in row["by_position"]
        )
        log(
            f"  {name:34s} {row['test']['J']:.4f} ({row['test']['PPL']:6.2f}) | "
            f"{row['valid_J']:.4f} | {groups}"
        )
    return output


# ---------------------------------------------------------------------------
# Stage 2: costs (D1.7)


def training_cost(record: dict[str, Any]) -> dict[str, Any]:
    summary = record["summary"]
    keys = (
        "seconds",
        "train_seconds",
        "eval_seconds",
        "seconds_to_best",
        "train_seconds_to_best",
        "steps",
        "best_step",
        "epochs_done",
        "targets_seen",
        "stopped_early",
    )
    return {key: summary.get(key) for key in keys} | {
        "peak_lr": summary["config"]["peak_lr"],
        "planned_epochs": summary["config"]["epochs"],
        "batch_tokens": summary["config"]["batch_tokens"],
        "promoted_from": record.get("promoted_from"),  # the final run is this try, not extra
    }


def tuning_cost(ctx: Context, model_name: str) -> dict[str, Any]:
    tries = sorted(ctx.runs.glob(f"{model_name}-try-lr*/run.json"))
    rows = []
    for path in tries:
        summary = json.loads(path.read_text())["summary"]
        rows.append(
            {
                "run": path.parent.name,
                "peak_lr": summary["config"]["peak_lr"],
                "best_valid_J": summary["best_valid_J"],
                "seconds": summary["seconds"],
                "train_seconds": summary["train_seconds"],
            }
        )
    final = load_record(ctx, f"{model_name}-full")
    promoted = final.get("promoted_from")
    extra = {"seconds": 0.0, "train_seconds": 0.0}
    if promoted not in {r["run"] for r in rows}:  # a final run trained after the tries
        extra = {k: float(final["summary"][k]) for k in extra}
    return {
        "tries": rows,
        "count": len(rows),
        "final_is_try": promoted,
        "seconds_tries": float(sum(r["seconds"] for r in rows)),
        "train_seconds_tries": float(sum(r["train_seconds"] for r in rows)),
        "total_seconds": float(sum(r["seconds"] for r in rows)) + extra["seconds"],
        "total_train_seconds": float(sum(r["train_seconds"] for r in rows))
        + extra["train_seconds"],
    }


def stage_costs(ctx: Context, test_scores: dict[str, Any] | None) -> dict[str, Any]:
    test = ctx.data["lm_test"]
    costs = ctx.config["costs"]
    repeats = int(costs["repeats"])
    context = mean_context(test)
    reference = (test_scores or {}).get("models", {})
    output: dict[str, Any] = {
        "workload": {
            "pile": "lm_test",
            "reviews": len(test),
            "targets": test.n_targets,
            "batch_tokens": ctx.batch_tokens,
            "repeats": repeats,
            "mean_context": context,
        },
        "training": {},
        "tuning": {},
        "models": {},
    }
    for name in final_models(ctx.config):
        output["training"][name] = training_cost(load_record(ctx, name))
    for model_name in costs["tries"]:
        output["tuning"][model_name] = tuning_cost(ctx, model_name)

    for name in dict.fromkeys(costs["gpu"] + costs["cpu"]):
        model, record = load_neural(ctx, name, torch.device("cpu"))
        output["models"][name] = {
            "label": describe(record["spec"]),
            "parameters": count_parameters(model),
            "checkpoint_bytes": (ctx.checkpoints / f"{name}.pt").stat().st_size,
            "flops_per_prediction": forward_flops(model, context),
            "speed": {},
        }
        del model

    for device_name, names in (("gpu", costs["gpu"]), ("cpu", costs["cpu"])):
        device = ctx.device if device_name == "gpu" else torch.device("cpu")
        for name in names:
            model, _ = load_neural(ctx, name, device)
            log(f"  timing {name} on {device} ({repeats} passes after a warm-up pass)")
            speed = scoring_speed(model, test, ctx.batch_tokens, device, repeats)
            expected = reference.get(name, {}).get("test", {}).get("J")
            speed["matches_test_J"] = (
                None if expected is None else abs(speed["J"] - expected) < 1e-3
            )
            output["models"][name]["speed"][device_name] = speed
            log(
                f"    {speed['targets_per_second']:,.0f} targets/s "
                f"(range {speed['range_targets_per_second'][0]:,.0f} to "
                f"{speed['range_targets_per_second'][1]:,.0f})"
            )
            free(model, device)

    log(f"  timing {NGRAM} on cpu")
    order = ctx.kn.order
    output["models"][NGRAM] = {
        "label": f"Kneser-Ney {order}-gram",
        "stored_ngrams": ctx.table.stored_entries(order),
        "table_bytes": ctx.table.nbytes(order),
        "counting_seconds": ctx.counting_seconds,
        "counting_seconds_recorded": ctx.ngram_record.get("counting_seconds"),
        "peak_memory_bytes_recorded": ctx.ngram_record.get("peak_memory_bytes"),
        "speed": {"cpu": ngram_scoring_speed(ctx.kn, test, repeats)},
    }
    embeddings = Path(ctx.config["embeddings_results"])
    if embeddings.exists():
        skipgram = json.loads(embeddings.read_text()).get("skipgram", {})
        output["skipgram"] = {
            k: skipgram.get(k) for k in ("seconds", "cpu_seconds", "threads", "tokens_processed")
        }
    output["peak_process_memory_bytes"] = peak_process_memory()
    save(ctx, "costs.json", output)

    log("\nModel | core params | GFLOPs/prediction | GPU targets/s | CPU targets/s | train min")
    for name, row in output["models"].items():
        gpu = row["speed"].get("gpu", {}).get("targets_per_second")
        cpu = row["speed"].get("cpu", {}).get("targets_per_second")
        core = row.get("parameters", {}).get("core")
        flops = row.get("flops_per_prediction", {}).get("total")
        minutes = output["training"].get(name, {}).get("train_seconds")
        log(
            f"  {name:22s} {core if core is not None else '-':>10} "
            f"{flops / 1e9 if flops else float('nan'):>8.5f} "
            f"{gpu if gpu else float('nan'):>14,.0f} {cpu if cpu else float('nan'):>14,.0f} "
            f"{minutes / 60 if minutes else float('nan'):>8.1f}"
        )
    return output


# ---------------------------------------------------------------------------
# Stage 3: generated text (D1.6)


def openings(
    reviews: EncodedReviews, prompt_tokens: int, length: int, count: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random reviews long enough for a prompt and a full continuation.

    Returns the prompts (<bos> plus the first prompt_tokens tokens), the real
    continuations (the next `length` tokens) and the chosen review positions.
    """
    eligible = np.nonzero(reviews.lengths >= prompt_tokens + length)[0]
    if eligible.size < count:
        raise ValueError(f"Only {eligible.size} reviews have {prompt_tokens + length}+ tokens")
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(eligible, size=count, replace=False))
    texts = np.stack([reviews.review(int(i))[: prompt_tokens + length] for i in chosen])
    texts = texts.astype(np.int64)
    prompts = np.concatenate(
        [np.full((count, 1), BOS_ID, dtype=np.int64), texts[:, :prompt_tokens]], 1
    )
    return prompts, texts[:, prompt_tokens:], chosen


@torch.no_grad()
def real_nll(
    model: nn.Module | KneserNey, prompts: np.ndarray, real: np.ndarray, device
) -> np.ndarray:
    """-log P of every real continuation token, given the real text before it."""
    full = np.concatenate([prompts, real], axis=1)  # <bos>, prompt, continuation
    start = prompts.shape[1] - 1  # target column of the first continuation token
    if isinstance(model, KneserNey):
        reviews = EncodedReviews.from_lists([row[1:].tolist() for row in full])
        log_probs = model.log_probs(reviews).reshape(full.shape[0], full.shape[1])  # + <eos>
        return -log_probs[:, start : start + real.shape[1]]
    model.eval()
    parts = []
    for first in range(0, full.shape[0], REAL_CHUNK):
        chunk = torch.as_tensor(full[first : first + REAL_CHUNK], dtype=torch.long, device=device)
        log_p = torch.log_softmax(model(chunk[:, :-1]).float(), dim=-1)
        picked = log_p.gather(2, chunk[:, 1:, None]).squeeze(2)
        parts.append(-picked[:, start:].cpu().double().numpy())
    return np.concatenate(parts)


def encode_prompts(vocab: Vocabulary, gen: dict[str, Any]) -> list[list[int]]:
    """The prompts as token ids after <bos>; a word outside the vocabulary stops the script."""
    encoded = []
    for index, prompt in enumerate(gen["prompts"], 1):
        tokens = prompt["text"].split()
        ids = vocab.encode(tokens)
        if UNK_ID in ids:
            outside = [t for t, i in zip(tokens, ids, strict=True) if i == UNK_ID]
            raise SystemExit(f"Prompt {index} has words outside the vocabulary: {outside}")
        encoded.append([BOS_ID, *ids])
    return encoded


def temperature_grid(start: float, stop: float, step: float) -> list[float]:
    count = int(round((stop - start) / step)) + 1
    return [round(start + i * step, 4) for i in range(count)]


def stage_generation(ctx: Context, gen: dict[str, Any]) -> dict[str, Any]:
    train, valid, test = ctx.data["lm_train"], ctx.data["lm_valid"], ctx.data["lm_test"]
    seed = int(gen["seed"])
    if gen["length"] != "median_train_review":
        raise ValueError("length must be median_train_review (D1.6)")
    length = int(math.floor(float(np.median(train.lengths)) + 0.5))
    log(f"  new tokens per sample: {length} (median review length of lm_train)")

    # The temperature test, on validation reviews only.
    test_cfg = gen["temperature_test"]
    k = int(test_cfg["prompt_tokens"])
    prompts, real, _ = openings(valid, k, length, int(test_cfg["reviews"]), seed)
    grid_cfg = test_cfg["grid"]
    step = float(grid_cfg["step"])
    low_limit, high_limit = (float(x) for x in test_cfg["limits"])
    grid = temperature_grid(float(grid_cfg["start"]), float(grid_cfg["stop"]), step)
    measured: dict[str, dict[float, dict[str, float]]] = {m: {} for m in test_cfg["models"]}
    real_j: dict[str, float] = {}
    while True:
        for name in test_cfg["models"]:
            missing = [t for t in grid if t not in measured[name]]
            if not missing:
                continue
            model = load_model(ctx, name)
            if name not in real_j:
                real_j[name] = float(real_nll(model, prompts, real, ctx.device).mean())
            for temperature in missing:
                written, own = generate(
                    make_sampler(model, ctx.device), prompts, length, temperature, seed
                )
                scores = text_scores(ctx.table, prompts, written)
                measured[name][temperature] = {
                    "own_J": float(-own.mean()),
                    "gap": float(-own.mean()) - real_j[name],
                    "seq_rep_4": scores["seq_rep_4"],
                    "distinct_2": scores["distinct_2"],
                }
            log(
                f"    {name}: real-text J {real_j[name]:.3f}; "
                f"measured T = {missing[0]} to {missing[-1]}"
            )
            if name != NGRAM:
                free(model, ctx.device)
        gaps = {name: [measured[name][t]["gap"] for t in grid] for name in test_cfg["models"]}
        choice = choose_temperature(grid, gaps)
        if choice["at_low_edge"] and grid[0] - step >= low_limit - 1e-9:
            grid = [round(grid[0] - step, 4), *grid]
        elif choice["at_high_edge"] and grid[-1] + step <= high_limit + 1e-9:
            grid = [*grid, round(grid[-1] + step, 4)]
        else:
            break
    temperature = float(choice["temperature"])
    log(f"  chosen T = {temperature} (mean |gap| {choice['mean_abs_gap']:.3f} nats)")
    log(f"  each model's own T: {choice['own_temperature']}")

    # Scores at the chosen T, on test openings, beside the real continuations.
    score_cfg = gen["scores"]
    k_test = int(score_cfg["prompt_tokens"])
    test_prompts, test_real, _ = openings(test, k_test, length, int(score_cfg["reviews"]), seed)
    resamples = int(score_cfg["bootstrap"])
    scores: dict[str, Any] = {
        "real": {
            "scores": text_scores(ctx.table, test_prompts, test_real),
            "interval_95": bootstrap_scores(ctx.table, test_prompts, test_real, resamples, seed),
            "distinct_halves": distinct_halves(test_real),
        }
    }
    examples: list[str] = [f"Generated text (D1.6), T = {temperature}, {length} new tokens\n"]
    prompt_results: dict[str, Any] = {}
    prompt_ids = encode_prompts(ctx.vocab, gen)
    unigram = dict(zip(ctx.table.keys[1].tolist(), ctx.table.counts[1].tolist(), strict=True))
    for index, prompt in enumerate(gen["prompts"], 1):
        tokens = prompt["text"].split()
        ids = prompt_ids[index - 1][1:]
        prompt_results[str(index)] = {
            "name": prompt["name"],
            "text": prompt["text"],
            "training_counts": {
                t: int(unigram.get(i, 0)) for t, i in zip(tokens, ids, strict=True)
            },
            "models": {},
        }
    samples = int(gen["samples_per_prompt"])
    for name in gen["models"]:
        model = load_model(ctx, name)
        written, own = generate(
            make_sampler(model, ctx.device), test_prompts, length, temperature, seed
        )
        nll_real = float(real_nll(model, test_prompts, test_real, ctx.device).mean())
        scores[name] = {
            "scores": text_scores(ctx.table, test_prompts, written),
            "interval_95": bootstrap_scores(ctx.table, test_prompts, written, resamples, seed),
            "distinct_halves": distinct_halves(written),
            "own_J": float(-own.mean()),
            "real_J": nll_real,
            "gap": float(-own.mean()) - nll_real,
        }
        for index, ids in enumerate(prompt_ids, 1):
            batch = np.tile(np.asarray(ids, dtype=np.int64), (samples, 1))
            sample_tokens, sample_own = generate(
                make_sampler(model, ctx.device), batch, length, temperature, seed + index
            )
            middle = middle_sample(sample_own)
            ppl = np.exp(-sample_own.mean(axis=1))
            prompt_results[str(index)]["models"][name] = {
                "perplexities": ppl.round(3).tolist(),
                "middle": middle,
                "middle_perplexity": float(ppl[middle]),
                "middle_scores": text_scores(
                    ctx.table, batch[middle : middle + 1], sample_tokens[middle : middle + 1]
                ),
            }
            prompt_words = ctx.vocab.decode(ids[1:])
            texts = [
                detokenize(prompt_words + ctx.vocab.decode(row)) for row in sample_tokens.tolist()
            ]
            examples.append(f"\n[prompt {index}: {gen['prompts'][index - 1]['name']}] [{name}]")
            for i, text in enumerate(texts):
                mark = ">>" if i == middle else "  "
                examples.append(f"{mark} ppl {ppl[i]:7.2f}  {text}")
        log(f"    {name}: written-text J {scores[name]['own_J']:.3f}, real-text J {nll_real:.3f}")
        if name != NGRAM:
            free(model, ctx.device)

    output = {
        "length": length,
        "length_rule": "median review length of lm_train",
        "temperature": temperature,
        "temperature_test": {
            "pile": test_cfg["pile"],
            "reviews": int(test_cfg["reviews"]),
            "prompt_tokens": k,
            "grid": grid,
            "real_J": real_j,
            "measured": {n: {str(t): v for t, v in m.items()} for n, m in measured.items()},
            "choice": choice,
        },
        "scores": scores,
        "scores_setup": {
            "pile": score_cfg["pile"],
            "reviews": int(score_cfg["reviews"]),
            "prompt_tokens": k_test,
        },
        "prompts": prompt_results,
        "samples_per_prompt": samples,
        "report_body_prompts": gen.get("report_body_prompts"),
        "seed": seed,
    }
    save(ctx, "generation.json", output)
    ctx.results.mkdir(parents=True, exist_ok=True)
    (ctx.results / "examples_generation.txt").write_text("\n".join(examples) + "\n")

    log(
        f"\nScores at the chosen T ({int(score_cfg['reviews'])} test openings), "
        "real continuations first"
    )
    header = "seq-rep-2 seq-rep-4 distinct-1 distinct-2 copied-5gram unk"
    log(f"  {'':34s} {header}")
    keys = ("seq_rep_2", "seq_rep_4", "distinct_1", "distinct_2", "copied_5gram_share", "unk_share")
    for name, row in scores.items():
        values = " ".join(f"{row['scores'][k]:9.3f}" for k in keys)
        log(f"  {name:34s} {values}")
    log(
        "\nThe samples are in results/eval/examples_generation.txt (the middle one of each "
        "prompt and model is marked >>). They are not printed here, because logs are kept "
        "in git and the text may repeat training reviews."
    )
    return output


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    parser.add_argument("--generation", type=Path, default=Path("configs/generation.yaml"))
    parser.add_argument("--stages", default="test,costs,generation")
    parser.add_argument("--device", help="override the config's device (tests use cpu)")
    args = parser.parse_args(argv)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = set(stages) - {"test", "costs", "generation"}
    if unknown:
        parser.error(f"unknown stages: {sorted(unknown)}")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    gen = yaml.safe_load(args.generation.read_text(encoding="utf-8"))
    device = pick_device(args.device or config.get("device"))
    vocab = Vocabulary.load(Path(config["vocabulary"]))
    encode_prompts(vocab, gen)  # stop now, not after an hour, if a prompt word is unknown
    piles = load_piles(Path(config["piles"]))
    started = time.perf_counter()
    data = encode_piles(
        Path(config["corpus"]), {k: piles[k] for k in ("lm_train", "lm_valid", "lm_test")}, vocab
    )
    log(
        f"|V| = {len(vocab):,}; read the corpus in {time.perf_counter() - started:.0f} s; "
        f"lm_test {len(data['lm_test']):,} reviews, {data['lm_test'].n_targets:,} targets; "
        f"device {device}"
    )

    ngram_record = json.loads(Path(config["ngram_results"]).read_text())
    kn_order = int(ngram_record["chosen"]["kneser_ney"]["order"])
    addk_choice = ngram_record["chosen"]["add_k"]
    started = time.perf_counter()
    table = NgramCounts.build(data["lm_train"], len(vocab), max(5, kn_order, addk_choice["order"]))
    counting_seconds = time.perf_counter() - started
    kn = KneserNey(table, kn_order)
    addk = AddK(table, int(addk_choice["order"]), float(addk_choice["k"]))
    addk.fit_weights(data["lm_valid"])  # lambda by EM on validation, as in train_ngram.py
    log(
        f"Counted n-grams in {counting_seconds:.0f} s; Kneser-Ney order {kn_order}, "
        f"add-k {addk_choice}"
    )

    ctx = Context(
        config=config,
        vocab=vocab,
        data=data,
        table=table,
        kn=kn,
        addk=addk,
        ngram_record=ngram_record,
        counting_seconds=counting_seconds,
        device=device,
        runs=Path(config["runs"]),
        checkpoints=Path(config["checkpoints"]),
        results=Path(config["results"]),
    )
    log("\n== Checks before any test data is used ==")
    ctx.checks = check_models(ctx)
    test_scores = None
    if "test" in stages:
        log("\n== Test scores and loss by position (D1.5) ==")
        test_scores = stage_test(ctx)
    elif (ctx.results / "test.json").exists():
        test_scores = json.loads((ctx.results / "test.json").read_text())
    if "costs" in stages:
        log("\n== Costs (D1.7) ==")
        stage_costs(ctx, test_scores)
    if "generation" in stages:
        log("\n== Generated text (D1.6) ==")
        stage_generation(ctx, gen)
    log(f"\nDone. Results in {ctx.results}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
