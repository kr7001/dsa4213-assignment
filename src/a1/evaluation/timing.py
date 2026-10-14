"""Speed and memory measurements.

GPU work runs in the background, so every timer waits for the GPU to finish
first (synchronize). Each measurement runs its batches once untimed, so the GPU
has prepared a program for every batch shape, then times a second pass over
the same batches. Speed is reported in targets per second: real guesses,
padding excluded.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from a1.data.sequences import EncodedReviews, collate, make_batches
from a1.data.vocab import PAD_ID


def pick_device(name: str | None = None) -> torch.device:
    """The requested device, or Apple's GPU (mps), then CUDA, then the CPU."""
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_in_use(device: torch.device) -> int | None:
    """Bytes the GPU driver holds for this process (MPS) or the peak allocated (CUDA)."""
    if device.type == "mps":
        return int(torch.mps.driver_allocated_memory())
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return None


def release_memory(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def is_out_of_memory(error: BaseException) -> bool:
    text = str(error).lower()
    return "out of memory" in text or "insufficient memory" in text


def training_speed(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    reviews: EncodedReviews,
    batch_tokens: int,
    device: torch.device,
    steps: int,
    seed: int,
    clip_norm: float = 1.0,
) -> dict[str, Any]:
    """Time full training steps (forward, backward, clipping, update) at one batch size.

    The same batches run twice: the first pass lets the GPU prepare its programs
    for every batch shape, and only the second pass is timed, so the result is
    the speed of a long run, not of its first minutes.
    """
    batches = make_batches(reviews.lengths, batch_tokens, seed=seed)
    if len(batches) < steps:
        raise ValueError(f"Only {len(batches)} batches; need {steps}")
    batches = batches[:steps]
    model.train()

    def one_step(index: np.ndarray) -> int:
        inputs, targets = collate(reviews, index, device)
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=PAD_ID
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        return int(inputs.numel())

    for index in batches:  # untimed pass
        one_step(index)
    synchronize(device)
    start = time.perf_counter()
    positions = sum(one_step(index) for index in batches)
    synchronize(device)
    seconds = time.perf_counter() - start
    targets = sum(int(reviews.lengths[index].sum()) + index.size for index in batches)
    return {
        "batch_tokens": batch_tokens,
        "steps_timed": steps,
        "seconds_per_step": seconds / steps,
        "targets_per_second": targets / seconds,
        "targets_per_step": targets / steps,
        "padding_share": 1.0 - targets / positions,
        "memory_bytes": memory_in_use(device),
    }


@torch.no_grad()
def scoring_speed(
    model: nn.Module,
    reviews: EncodedReviews,
    batch_tokens: int,
    device: torch.device,
    repeats: int = 5,
) -> dict[str, Any]:
    """Speed and memory on the common workload: scoring every review once (D1.7).

    One untimed pass first (the GPU prepares its programs for each batch shape),
    then `repeats` timed passes over the same batches as the test score. Each
    pass sums the loss on the device and reads it once at the end; the GPU is
    synchronised before every clock reading, so the time covers the finished
    work, not only the launch. Speed is targets per second at the median pass.
    """
    model.eval()
    batches = make_batches(reviews.lengths, batch_tokens)
    targets = reviews.n_targets

    def one_pass() -> float:
        total = torch.zeros((), device=device)
        for index in batches:
            inputs, labels = collate(reviews, index, device)
            logits = model(inputs)
            total = total + F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=PAD_ID,
                reduction="sum",
            )
        return float(total) / targets

    release_memory(device)
    warm_j = one_pass()
    seconds = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        one_pass()
        synchronize(device)
        seconds.append(time.perf_counter() - start)
    median = float(np.median(seconds))
    return {
        "device": str(device),
        "threads": torch.get_num_threads() if device.type == "cpu" else None,
        "batch_tokens": batch_tokens,
        "targets": targets,
        "J": warm_j,
        "seconds": seconds,
        "median_seconds": median,
        "targets_per_second": targets / median,
        "range_targets_per_second": [targets / max(seconds), targets / min(seconds)],
        "memory_bytes": memory_in_use(device),
    }


def ngram_scoring_speed(model: Any, reviews: EncodedReviews, repeats: int = 5) -> dict[str, Any]:
    """The same workload for a count model on the CPU (one untimed pass, then timed passes)."""
    log_probs = model.log_probs(reviews)
    seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.log_probs(reviews)
        seconds.append(time.perf_counter() - start)
    median = float(np.median(seconds))
    targets = int(log_probs.size)
    return {
        "device": "cpu",
        "targets": targets,
        "J": float(-log_probs.mean()),
        "seconds": seconds,
        "median_seconds": median,
        "targets_per_second": targets / median,
        "range_targets_per_second": [targets / max(seconds), targets / min(seconds)],
    }
