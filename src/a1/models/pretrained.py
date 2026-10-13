"""Put a ready-made input table into a model, and check what training did to it (Part II).

Part II compares input tables E in one architecture: random and trainable (the
Part I model), our own skip-gram vectors kept fixed, and public GloVe vectors
kept fixed. "In the fixed settings, keep the entire input-embedding table
fixed" (assignment, Part II), so freezing covers every row: the words GloVe
lacks and the special tokens too.

A fixed table gets requires_grad = False, so it has no gradient, the optimizer
never sees it (trainer.make_optimizer skips it) and weight decay cannot touch
it. Every run also compares a fingerprint (SHA-256 of the raw float32 bytes) of
E before and after training, and train_lm.py stops if a fixed table changed.

The model is built first, as in Part I, and the table is copied in afterwards.
Building draws E's random start before the LSTM and the output head, so for a
given seed every setting starts the LSTM and the output head from the same
numbers and reads the training batches in the same order; only E differs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from a1.data.vocab import PAD_ID


def table_sha256(table: torch.Tensor | np.ndarray) -> str:
    """Fingerprint of a table's values: SHA-256 of its float32 bytes."""
    if isinstance(table, torch.Tensor):
        table = table.detach().to("cpu", torch.float32).contiguous().numpy()
    return hashlib.sha256(np.ascontiguousarray(table, dtype=np.float32).tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    """Fingerprint of a file on disk."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_table(path: Path, vocab_size: int, dim: int) -> np.ndarray:
    """A saved |V| x d table, checked for shape, finite values and a zero <pad> row."""
    table = np.load(path)
    if table.shape != (vocab_size, dim):
        raise ValueError(f"{path} has shape {table.shape}; expected ({vocab_size}, {dim})")
    if not np.isfinite(table).all():
        raise ValueError(f"{path} has values that are not finite")
    if np.any(table[PAD_ID] != 0):
        raise ValueError(f"{path}: the <pad> row must be zero")
    return table.astype(np.float32, copy=False)


def install_table(model: nn.Module, table: np.ndarray | None, freeze: bool) -> None:
    """Copy a table into the model's E (if given) and freeze all of E (if asked).

    With table=None, E keeps the random start it was built with.
    """
    weight = model.embedding.weight
    if table is not None:
        if tuple(table.shape) != tuple(weight.shape):
            raise ValueError(f"Table shape {table.shape} does not match E {tuple(weight.shape)}")
        with torch.no_grad():
            weight.copy_(torch.as_tensor(table, dtype=weight.dtype))
    weight.requires_grad_(not freeze)


def embedding_change(start: torch.Tensor, end: torch.Tensor) -> dict[str, Any]:
    """How far E moved during training: unchanged or not, and by how much.

    relative_change is ||E_end - E_start|| / ||E_start|| (Frobenius norms), so
    0 means untouched; median_row_cosine compares each word's start and end
    vectors (the zero <pad> row is left out).
    """
    start = start.detach().to("cpu", torch.float64)
    end = end.detach().to("cpu", torch.float64)
    rows = torch.ones(start.shape[0], dtype=torch.bool)
    rows[PAD_ID] = False
    cosine = torch.nn.functional.cosine_similarity(start[rows], end[rows], dim=1)
    return {
        "unchanged": bool(torch.equal(start, end)),
        "relative_change": float((end - start).norm() / start.norm()),
        "median_row_cosine": float(cosine.median()),
    }
