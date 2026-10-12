"""One training run under the rules locked on 12 Oct (D0.6 part 1, Training rules tab).

- AdamW for every model: beta1 = 0.9, beta2 = 0.999, eps = 1e-8 (Kingma and Ba's
  suggested defaults) and decoupled weight decay 0.1 (Loshchilov and Hutter,
  2019; 0.1 as in GPT-3). Decay applies to weight matrices only: not to biases,
  LayerNorm gains or the input table E, so pretrained vectors in Part II are not
  pulled towards zero.
- One batch size in tokens for all models; the learning-rate plan of
  schedule.py; gradients clipped at a global norm of 1.0, with the unclipped
  norm of every step kept so the test run can check the threshold (Pascanu et
  al., 2013).
- Validation J four times per epoch; stop after three checks with no new best
  and keep the best checkpoint (stopping.py).
- Only the peak learning rate is tuned, per model, over the same three values.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from a1.data.sequences import EncodedReviews, collate, make_batches
from a1.data.vocab import PAD_ID
from a1.evaluation.perplexity import evaluate
from a1.training.schedule import FINAL_FRACTION, WARMUP_FRACTION, make_scheduler
from a1.training.stopping import PATIENCE, EarlyStopping

SEED = 4213
LEARNING_RATES = (3e-4, 1e-3, 3e-3)  # the same three tries for every model


@dataclass
class TrainConfig:
    peak_lr: float = 1e-3
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    batch_tokens: int = 8192
    epochs: float = 1.0  # planned length; the learning-rate plan ends exactly here
    max_steps: int | None = None  # cut the plan to this many steps (short runs)
    warmup_fraction: float = WARMUP_FRACTION
    final_fraction: float = FINAL_FRACTION
    clip_norm: float = 1.0
    checks_per_epoch: int = 4
    patience: int = PATIENCE
    seed: int = SEED
    log_every: int = 50


def make_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.AdamW:
    """AdamW with weight decay on weight matrices only (not biases, norms or E)."""
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.startswith("embedding."):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.peak_lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
    )


def gradient_norm_summary(norms: list[float], clip_norm: float) -> dict[str, float]:
    """How large the unclipped gradients were, and how often clipping fired.

    Pascanu et al. (2013) set the threshold from "statistics on the average norm
    over a sufficiently large number of updates", and report that thresholds from
    half to ten times that average still converge. Early steps have unusually
    large gradients, so the same figures are also given for the second half of
    the steps ("late_").
    """
    values = np.asarray(norms, dtype=float)
    if values.size == 0:
        return {}

    def figures(part: np.ndarray, prefix: str) -> dict[str, float]:
        mean = float(part.mean())
        return {
            f"{prefix}mean": mean,
            f"{prefix}median": float(np.median(part)),
            f"{prefix}p90": float(np.percentile(part, 90)),
            f"{prefix}p99": float(np.percentile(part, 99)),
            f"{prefix}max": float(part.max()),
            f"{prefix}share_clipped": float((part > clip_norm).mean()),
            f"{prefix}threshold_in_safe_range": bool(0.5 * mean <= clip_norm <= 10 * mean),
        }

    return (
        {"steps": int(values.size)}
        | figures(values, "")
        | figures(values[values.size // 2 :], "late_")
    )


def train(
    model: nn.Module,
    train_data: EncodedReviews,
    valid_data: EncodedReviews | None,
    config: TrainConfig,
    device: torch.device,
    checkpoint: Path | None = None,
    after_step: Callable[[int, nn.Module], None] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Train with the locked rules; return the history and a summary.

    Args:
        model: A fresh model; it ends holding the best checkpoint's weights.
        train_data, valid_data: Encoded reviews; without validation data, no
            checks or early stopping happen (the test run's short runs).
        config: The settings above.
        device: Where to train.
        checkpoint: If given, the best weights are also saved there.
        after_step: Called as after_step(step, model) after every update.
        log: Where progress lines go.
    """
    torch.manual_seed(config.seed)
    model.to(device)
    optimizer = make_optimizer(model, config)
    first_epoch = make_batches(train_data.lengths, config.batch_tokens, seed=config.seed)
    steps_per_epoch = len(first_epoch)
    total_steps = max(1, round(steps_per_epoch * config.epochs))
    if config.max_steps is not None:
        total_steps = min(total_steps, config.max_steps)
    scheduler = make_scheduler(
        optimizer, total_steps, config.warmup_fraction, config.final_fraction
    )
    check_every = max(1, steps_per_epoch // config.checks_per_epoch)
    stopper = EarlyStopping(config.patience)
    history: list[dict[str, Any]] = []
    norms: list[torch.Tensor] = []
    running_loss = torch.zeros((), device=device)
    running_targets = 0
    best_state: dict[str, torch.Tensor] | None = None
    seconds_to_best = 0.0
    step = targets_seen = epoch = 0
    diverged = False
    start = time.perf_counter()

    while step < total_steps and not stopper.should_stop and not diverged:
        batches = (
            first_epoch
            if epoch == 0
            else make_batches(train_data.lengths, config.batch_tokens, seed=config.seed + epoch)
        )
        for index in batches:
            if step >= total_steps or stopper.should_stop:
                break
            model.train()
            inputs, targets = collate(train_data, index, device)
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=PAD_ID
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norms.append(torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm))
            optimizer.step()
            scheduler.step()
            n = int(train_data.lengths[index].sum()) + index.size
            running_loss += loss.detach() * n
            running_targets += n
            targets_seen += n
            step += 1
            if after_step is not None:
                after_step(step, model)
            if step % config.log_every == 0 or step == total_steps:
                train_j = float(running_loss) / running_targets
                if not math.isfinite(train_j):
                    log(f"step {step}: the training loss is not finite; stopping")
                    diverged = True
                    break
                history.append(
                    {
                        "step": step,
                        "targets_seen": targets_seen,
                        "seconds": time.perf_counter() - start,
                        "lr": scheduler.get_last_lr()[0],
                        "train_J": train_j,
                    }
                )
                running_loss.zero_()
                running_targets = 0
            if valid_data is not None and (step % check_every == 0 or step == total_steps):
                result = evaluate(model, valid_data, config.batch_tokens, device)
                improved = stopper.update(result["J"], step)
                elapsed = time.perf_counter() - start
                history.append(
                    {
                        "step": step,
                        "targets_seen": targets_seen,
                        "seconds": elapsed,
                        "valid_J": result["J"],
                        "valid_PPL": result["PPL"],
                        "new_best": improved,
                    }
                )
                log(
                    f"step {step}/{total_steps}  valid J {result['J']:.4f}  "
                    f"PPL {result['PPL']:.1f}{'  (best)' if improved else ''}"
                )
                if improved:
                    seconds_to_best = elapsed
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    }
                    if checkpoint is not None:
                        checkpoint.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(best_state, checkpoint)
        epoch += 1

    seconds = time.perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)
    norm_values = [float(v) for v in torch.stack(norms).cpu()] if norms else []
    return {
        "config": asdict(config),
        "steps": step,
        "planned_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "epochs_done": step / steps_per_epoch,
        "targets_seen": targets_seen,
        "seconds": seconds,
        "seconds_to_best": seconds_to_best,
        "stopped_early": stopper.should_stop,
        "diverged": diverged,
        "best_valid_J": None if best_state is None else stopper.best,
        "best_step": stopper.best_step,
        "gradient_norms": gradient_norm_summary(norm_values, config.clip_norm),
        "history": history,
    }
