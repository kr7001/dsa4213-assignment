"""Writing text with the language models and scoring what they write (D1.6, locked 14 Oct).

Every model writes with temperature sampling (Lecture 3, slide 21):

    q_{t,T}(w) = p_t(w)^{1/T} / sum_u p_t(u)^{1/T},

with one T for all models. <pad> and <bos> are never written, and <eos> is
blocked, so every sample has the same number of new tokens; <unk> may be
written and is counted. Each step draws one uniform number per sample from a
seeded stream and picks the first word whose cumulative probability reaches
it, so every model (and every T) uses the same random numbers.

T is measured, not guessed (choose_temperature): on validation reviews, the
temperature at which a model's perplexity on its own continuations matches its
perplexity on the real ones (Holtzman et al., 2020, section 4.2), averaged over
the Part I models.

The scores beside each model are also computed on the real continuations:

- repetition inside a sample: seq-rep-n = 1 - distinct n-grams / n-grams
  (Welleck et al., 2020, equation 10);
- variety across samples: distinct-n = distinct n-grams / tokens written
  (Li et al., 2016, Table 2);
- copying: the share of written 5-token windows that occur in the training
  reviews (novelty against the training set, as in McCoy et al.);
- the share of <unk>.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import torch
from torch import nn

from a1.data.vocab import BOS_ID, EOS_ID, PAD_ID, UNK_ID
from a1.models.ngram import KneserNey, NgramCounts, _lookup
from a1.models.recurrent import RecurrentLM
from a1.models.transformer import TransformerLM

NEVER_WRITTEN = (PAD_ID, BOS_ID, EOS_ID)


class Sampler(Protocol):
    """Next-token log-probabilities for a batch of growing texts."""

    def start(self, prompts: np.ndarray) -> np.ndarray:
        """Read the prompts (B, L), each starting with <bos>; return log P(next) as (B, |V|)."""
        ...

    def advance(self, tokens: np.ndarray) -> np.ndarray:
        """Append one token per text (B,); return log P(next) as (B, |V|)."""
        ...


def _log_softmax(logits: torch.Tensor) -> np.ndarray:
    # MPS has no float64: normalise in float32 on the device, then widen on the CPU.
    return torch.log_softmax(logits.float(), dim=-1).cpu().double().numpy()


class RecurrentSampler:
    """RNN or LSTM: the state h^(t) (and c^(t)) is carried forward, one token per step."""

    def __init__(self, model: RecurrentLM, device: torch.device) -> None:
        self.model = model.eval()
        self.device = device
        self.state = None

    @torch.no_grad()
    def _run(self, tokens: np.ndarray) -> np.ndarray:
        inputs = torch.as_tensor(tokens, dtype=torch.long, device=self.device)
        states, self.state = self.model.rnn(self.model.embedding(inputs), self.state)
        return _log_softmax(self.model.head(states[:, -1]))

    def start(self, prompts: np.ndarray) -> np.ndarray:
        self.state = None
        return self._run(prompts)

    def advance(self, tokens: np.ndarray) -> np.ndarray:
        return self._run(np.asarray(tokens)[:, None])


class TransformerSampler:
    """Transformer: the whole text so far is read again at every step (no cache)."""

    def __init__(self, model: TransformerLM, device: torch.device) -> None:
        self.model = model.eval()
        self.device = device
        self.prefix: torch.Tensor | None = None

    @torch.no_grad()
    def _run(self) -> np.ndarray:
        assert self.prefix is not None
        return _log_softmax(self.model.head(self.model.hidden_states(self.prefix)[:, -1]))

    def start(self, prompts: np.ndarray) -> np.ndarray:
        self.prefix = torch.as_tensor(prompts, dtype=torch.long, device=self.device)
        return self._run()

    def advance(self, tokens: np.ndarray) -> np.ndarray:
        assert self.prefix is not None
        new = torch.as_tensor(np.asarray(tokens)[:, None], dtype=torch.long, device=self.device)
        self.prefix = torch.cat([self.prefix, new], dim=1)
        return self._run()


class NgramSampler:
    """Kneser-Ney: each text keeps its own history; histories never reach back past <bos>."""

    def __init__(self, model: KneserNey) -> None:
        self.model = model
        self.histories: list[list[int]] = []

    def _run(self) -> np.ndarray:
        return np.log(np.stack([self.model.next_distribution(h) for h in self.histories]))

    def start(self, prompts: np.ndarray) -> np.ndarray:
        self.histories = [[int(t) for t in row] for row in np.asarray(prompts)]
        return self._run()

    def advance(self, tokens: np.ndarray) -> np.ndarray:
        for history, token in zip(self.histories, np.asarray(tokens).tolist(), strict=True):
            history.append(int(token))
        return self._run()


def make_sampler(model: nn.Module | KneserNey, device: torch.device) -> Sampler:
    if isinstance(model, KneserNey):
        return NgramSampler(model)
    if isinstance(model, RecurrentLM):
        return RecurrentSampler(model, device)
    if isinstance(model, TransformerLM):
        return TransformerSampler(model, device)
    raise TypeError(f"No sampler for {type(model).__name__}")


def temperature_probabilities(
    log_p: np.ndarray, temperature: float, blocked: Sequence[int] = NEVER_WRITTEN
) -> np.ndarray:
    """q_T = p^(1/T) / sum p^(1/T), with the blocked ids set to zero and the rest renormalised."""
    scores = np.array(log_p, dtype=np.float64) / temperature
    scores[:, list(blocked)] = -np.inf
    scores -= scores.max(axis=1, keepdims=True)
    q = np.exp(scores)
    return q / q.sum(axis=1, keepdims=True)


def generate(
    sampler: Sampler,
    prompts: np.ndarray,
    length: int,
    temperature: float,
    seed: int,
    blocked: Sequence[int] = NEVER_WRITTEN,
) -> tuple[np.ndarray, np.ndarray]:
    """Write `length` tokens after each prompt.

    Args:
        prompts: (B, L) token ids, each row starting with <bos>; all rows the same length.

    Returns:
        The written tokens (B, length), and each written token's log-probability under
        the model as trained (T = 1, nothing blocked): the numbers perplexity is made of.
    """
    prompts = np.asarray(prompts, dtype=np.int64)
    rows = prompts.shape[0]
    rng = np.random.default_rng(seed)
    tokens = np.empty((rows, length), dtype=np.int64)
    own = np.empty((rows, length), dtype=np.float64)
    log_p = sampler.start(prompts)
    for step in range(length):
        q = temperature_probabilities(log_p, temperature, blocked)
        cumulative = np.cumsum(q, axis=1)
        cumulative /= cumulative[:, -1:]
        draw = 1.0 - rng.random(rows)  # in (0, 1], so a word with probability 0 is never picked
        choice = (cumulative < draw[:, None]).sum(axis=1)
        tokens[:, step] = choice
        own[:, step] = log_p[np.arange(rows), choice]
        if step + 1 < length:
            log_p = sampler.advance(choice)
    return tokens, own


def middle_sample(own_log_probs: np.ndarray) -> int:
    """The middle sample when ranked by the model's perplexity on it (odd number of samples)."""
    nll = -np.asarray(own_log_probs).mean(axis=1)
    if nll.size % 2 == 0:
        raise ValueError("Use an odd number of samples so there is one middle sample")
    return int(np.argsort(nll, kind="stable")[nll.size // 2])


def repeated_share(tokens: np.ndarray, n: int) -> np.ndarray:
    """seq-rep-n of each sample: 1 - distinct n-grams / n-grams (Welleck et al., 2020)."""
    shares = []
    for row in np.asarray(tokens).tolist():
        grams = [tuple(row[i : i + n]) for i in range(len(row) - n + 1)]
        shares.append(1.0 - len(set(grams)) / len(grams) if grams else 0.0)
    return np.asarray(shares)


def distinct_share(tokens: np.ndarray, n: int) -> float:
    """distinct-n over all samples: distinct n-grams / tokens written (Li et al., 2016)."""
    tokens = np.asarray(tokens)
    grams: set[tuple[int, ...]] = set()
    for row in tokens.tolist():
        grams.update(tuple(row[i : i + n]) for i in range(len(row) - n + 1))
    return len(grams) / tokens.size


def seen_in_training(table: NgramCounts, texts: np.ndarray, first: int, n: int = 5) -> np.ndarray:
    """Whether the n tokens ending at each position from `first` on occur in the training reviews.

    Args:
        table: Counts of the training reviews, with windows of at least n tokens.
        texts: (B, L) token ids, each row starting with <bos>.
        first: The first column scored (the first written token).

    Returns:
        A float array (B, L - first): 1 if the window occurs in training, 0 if not,
        and NaN where no window of n tokens can end (fewer than n - 1 tokens since
        <bos>), so those positions are left out of every average.
    """
    if n > table.max_order:
        raise ValueError(f"The table only counts windows of up to {table.max_order} tokens")
    texts = np.asarray(texts, dtype=np.int64)
    rows, width = texts.shape
    flat = texts.reshape(-1)
    offsets = np.tile(np.arange(width, dtype=np.int64), rows)
    ranks = table.history_ranks(flat, offsets, n)
    seen = np.full(flat.size, np.nan)
    position = np.nonzero(offsets >= max(first, n - 1))[0]
    seen[position] = 0.0
    prefix = ranks[n - 1][position - 1]
    known = prefix >= 0
    position = position[known]
    keys = prefix[known] * table.vocab_size + flat[position]
    seen[position] = (_lookup(table.keys[n], keys) >= 0).astype(np.float64)
    return seen.reshape(rows, width)[:, first:]


def per_sample_scores(
    table: NgramCounts, prompts: np.ndarray, written: np.ndarray
) -> dict[str, np.ndarray]:
    """Scores that are sums over samples, kept per sample so they can be resampled.

    Copying and <unk> are kept as (count, positions) pairs, so a resampled share is
    a ratio of sums, exactly like the share over all samples.
    """
    written = np.asarray(written)
    seen = seen_in_training(table, np.concatenate([prompts, written], axis=1), prompts.shape[1])
    return {
        "seq_rep_2": repeated_share(written, 2),
        "seq_rep_4": repeated_share(written, 4),
        "copied": np.nansum(seen, axis=1),
        "windows": np.sum(~np.isnan(seen), axis=1).astype(np.float64),
        "unk": (written == UNK_ID).sum(axis=1).astype(np.float64),
        "tokens": np.full(written.shape[0], float(written.shape[1])),
    }


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
    total = float(denominator.sum())
    return float(numerator.sum()) / total if total else float("nan")


def text_scores(table: NgramCounts, prompts: np.ndarray, written: np.ndarray) -> dict[str, float]:
    """The D1.6 scores of one set of continuations (model-written or real)."""
    parts = per_sample_scores(table, prompts, written)
    return {
        "seq_rep_2": float(parts["seq_rep_2"].mean()),
        "seq_rep_4": float(parts["seq_rep_4"].mean()),
        "distinct_1": distinct_share(written, 1),
        "distinct_2": distinct_share(written, 2),
        "copied_5gram_share": _ratio(parts["copied"], parts["windows"]),
        "unk_share": _ratio(parts["unk"], parts["tokens"]),
    }


def bootstrap_scores(
    table: NgramCounts,
    prompts: np.ndarray,
    written: np.ndarray,
    resamples: int,
    seed: int,
) -> dict[str, list[float]]:
    """95% percentile intervals, resampling whole samples with replacement.

    Only for the scores that are averages over samples (repetition, copying,
    <unk>). distinct-n gets no bootstrap interval: a resample repeats some samples,
    which adds tokens but no new n-grams, so resampled distinct-n is biased low.
    distinct_halves gives a check of its stability instead.
    """
    parts = per_sample_scores(table, prompts, written)
    rng = np.random.default_rng(seed)
    rows = written.shape[0]
    draws: dict[str, list[float]] = {
        k: [] for k in ("seq_rep_2", "seq_rep_4", "copied_5gram_share", "unk_share")
    }
    for _ in range(resamples):
        pick = rng.integers(0, rows, rows)
        draws["seq_rep_2"].append(float(parts["seq_rep_2"][pick].mean()))
        draws["seq_rep_4"].append(float(parts["seq_rep_4"][pick].mean()))
        draws["copied_5gram_share"].append(_ratio(parts["copied"][pick], parts["windows"][pick]))
        draws["unk_share"].append(_ratio(parts["unk"][pick], parts["tokens"][pick]))
    return {
        key: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
        for key, values in draws.items()
    }


def distinct_halves(written: np.ndarray) -> dict[str, list[float]]:
    """distinct-1 and distinct-2 on the first and second half of the samples (no overlap)."""
    written = np.asarray(written)
    half = written.shape[0] // 2
    parts = (written[:half], written[half : 2 * half])
    return {f"distinct_{n}": [distinct_share(part, n) for part in parts] for n in (1, 2)}


def zero_crossing(grid: Sequence[float], gaps: Sequence[float]) -> float | None:
    """The T where a gap curve crosses zero, by linear interpolation (None if it never does)."""
    points = list(zip(grid, gaps, strict=True))
    for (t0, g0), (t1, g1) in zip(points, points[1:], strict=False):
        if g0 == 0:
            return float(t0)
        if g0 * g1 < 0:
            return float(t0 + (t1 - t0) * g0 / (g0 - g1))
    return float(points[-1][0]) if points and points[-1][1] == 0 else None


def choose_temperature(
    grid: Sequence[float], gaps: dict[str, Sequence[float]], resolution: float = 0.01
) -> dict[str, object]:
    """The shared T: smallest mean |gap| over the models, gaps interpolated between grid points.

    gap = perplexity on its own text minus perplexity on real text, in log units
    (mean NLL of written tokens minus mean NLL of the real tokens).

    Returns the chosen T (on the resolution grid), the mean |gap| there, each
    model's own zero crossing, and whether the choice sits at an end of the grid
    (then the grid should be extended, like the learning-rate search).
    """
    grid = [float(t) for t in grid]
    fine = np.round(np.arange(grid[0], grid[-1] + resolution / 2, resolution), 6)
    curves = np.stack([np.interp(fine, grid, np.asarray(g, dtype=float)) for g in gaps.values()])
    mean_abs = np.abs(curves).mean(axis=0)
    best = int(np.argmin(mean_abs))
    return {
        "temperature": float(fine[best]),
        "mean_abs_gap": float(mean_abs[best]),
        "own_temperature": {name: zero_crossing(grid, list(g)) for name, g in gaps.items()},
        "at_low_edge": best == 0,
        "at_high_edge": best == fine.size - 1,
    }


def detokenize(words: Sequence[str]) -> str:
    """Join tokens for reading: punctuation sticks to the word before it (Lecture3_Demo)."""
    text = ""
    for word in words:
        if word in {".", ",", "!", "?", ";", ":"}:
            text = text.rstrip() + word + " "
        else:
            text += word + " "
    return text.strip()
