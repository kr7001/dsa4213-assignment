"""Reviews as token-id sequences, grouped into piles and batches.

Piles. Every product has a random split key in [0, 1) (prepare.py), and a pile
is a range of keys, so all reviews of one product land in the same pile. The
final pile sizes are set after the test run. The test run only uses products
with a key below 0.5, and the training pile will start at 0 and cover at least
half of the products, so the test run never touches validation or test reviews.

Sequences. A review of n tokens becomes the input <bos> w1 ... wn and the
targets w1 ... wn <eos>: n + 1 guesses. Each review is scored on its own (D0.5).

Batches. Reviews are grouped so that a batch holds at most a fixed number of
positions, padding included: the batch size in tokens (D0.6). Reviews of
similar length go together so little padding is wasted. Training batches are
drawn from shuffled pools and come in a random order that depends on a seed;
evaluation batches are simply sorted by length. Each batch is padded up to
the next of 17 fixed widths (8, 16, 24, 32, 40, 48, 64, ... 448, 513), so the
GPU sees a few batch shapes instead of hundreds: Apple's GPU prepares a program
for every new shape, which made the first test run time that preparation
instead of the training. The widths grow by about a quarter each step, so a
batch is at most about a third padding; the padding is never scored.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from a1.data.vocab import BOS_ID, EOS_ID, PAD_ID, Vocabulary

POOL_BATCHES = 100  # a training pool holds about this many batches before sorting
# Batch widths in positions; 513 is the longest possible input: 512 tokens (D0.8) plus <bos>.
WIDTHS = (8, 16, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 448, 513)


def padded_width(longest: int, fixed: bool = True) -> int:
    """Positions in a batch whose longest review fills `longest` positions.

    With fixed=False the batch is as wide as its longest review, nothing more.
    """
    if fixed:
        for width in WIDTHS:
            if width >= longest:
                return width
    return longest


@dataclass(frozen=True)
class Pile:
    """The reviews of every product whose split key lies in [low, high)."""

    name: str
    low: float
    high: float

    def contains(self, key: float) -> bool:
        return self.low <= key < self.high


# Test-run piles: both inside the future training pile (see the module docstring).
PILOT_TRAIN = Pile("pilot_train", 0.0, 0.45)
PILOT_VALID = Pile("pilot_valid", 0.45, 0.5)
PILE_NAMES = ("lm_train", "lm_valid", "lm_test", "part3_train", "part3_valid", "part3_test")


def load_piles(path: Path) -> dict[str, Pile]:
    """Read the pile layout, a YAML map of name -> [low, high) split-key range.

    Checks the layout: known names, ranges inside [0, 1) that do not overlap,
    and a language-model training pile that starts at 0 and covers the test
    run's products (keys below 0.5), so the test run never saw other piles.
    """
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))["piles"]
    piles = {name: Pile(name, float(low), float(high)) for name, (low, high) in raw.items()}
    unknown = set(piles) - set(PILE_NAMES)
    if unknown:
        raise ValueError(f"Unknown piles: {sorted(unknown)}")
    ordered = sorted(piles.values(), key=lambda pile: pile.low)
    for pile in ordered:
        if not 0.0 <= pile.low < pile.high <= 1.0:
            raise ValueError(f"Bad range for {pile.name}: [{pile.low}, {pile.high})")
    for before, after in zip(ordered, ordered[1:], strict=False):
        if before.high > after.low:
            raise ValueError(f"{before.name} and {after.name} overlap")
    train = piles.get("lm_train")
    if train is None or train.low != 0.0 or train.high < PILOT_VALID.high:
        raise ValueError("lm_train must start at 0 and cover every key below 0.5")
    return piles


def iter_pile(path: Path, pile: Pile) -> Iterator[tuple[int, list[str]]]:
    """Yield (review id, tokens) for every review in the pile, in file order."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if pile.contains(row["split_key"]):
                yield row["id"], row["text"].split(" ")


def count_tokens(path: Path, pile: Pile) -> tuple[Counter, int]:
    """Token counts and the number of reviews in the pile."""
    counts: Counter = Counter()
    reviews = 0
    for _, tokens in iter_pile(path, pile):
        counts.update(tokens)
        reviews += 1
    return counts, reviews


@dataclass
class EncodedReviews:
    """Token ids of many reviews, stored back to back.

    Attributes:
        ids: Every review's token ids, without <bos> or <eos>.
        offsets: Review i is ids[offsets[i]:offsets[i + 1]].
        review_ids: The corpus id of each review.
    """

    ids: np.ndarray
    offsets: np.ndarray
    review_ids: np.ndarray

    @classmethod
    def from_lists(cls, reviews: list[list[int]]) -> EncodedReviews:
        """Build from one list of token ids per review (handy for tests)."""
        lengths = [len(ids) for ids in reviews]
        return cls(
            ids=np.asarray([i for ids in reviews for i in ids], dtype=np.int32),
            offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
            review_ids=np.arange(len(reviews), dtype=np.int64),
        )

    def __len__(self) -> int:
        return int(self.review_ids.size)

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    @property
    def n_targets(self) -> int:
        """Guesses over all reviews: every token plus one <eos> per review."""
        return int(self.ids.size + len(self))

    def review(self, i: int) -> np.ndarray:
        return self.ids[self.offsets[i] : self.offsets[i + 1]]

    def subset(self, indices: np.ndarray) -> EncodedReviews:
        """The reviews at these positions, in this order."""
        pieces = [self.review(int(i)) for i in indices]
        lengths = np.array([piece.size for piece in pieces], dtype=np.int64)
        ids = np.concatenate(pieces) if pieces else np.zeros(0, dtype=self.ids.dtype)
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        return EncodedReviews(ids, offsets, self.review_ids[np.asarray(indices, dtype=np.int64)])


def encode_piles(
    path: Path, piles: dict[str, Pile], vocab: Vocabulary
) -> dict[str, EncodedReviews]:
    """Read the corpus once and encode every pile."""
    parts: dict[str, tuple[list[int], list[int], list[int]]] = {
        name: ([], [0], []) for name in piles
    }
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for name, pile in piles.items():
                if pile.contains(row["split_key"]):
                    ids, offsets, review_ids = parts[name]
                    ids.extend(vocab.encode(row["text"].split(" ")))
                    offsets.append(len(ids))
                    review_ids.append(row["id"])
                    break
    return {
        name: EncodedReviews(
            ids=np.asarray(ids, dtype=np.int32),
            offsets=np.asarray(offsets, dtype=np.int64),
            review_ids=np.asarray(review_ids, dtype=np.int64),
        )
        for name, (ids, offsets, review_ids) in parts.items()
    }


def encode_pile(path: Path, pile: Pile, vocab: Vocabulary) -> EncodedReviews:
    """Read the pile and turn every review into token ids."""
    ids: list[int] = []
    offsets = [0]
    review_ids: list[int] = []
    for review_id, tokens in iter_pile(path, pile):
        ids.extend(vocab.encode(tokens))
        offsets.append(len(ids))
        review_ids.append(review_id)
    return EncodedReviews(
        ids=np.asarray(ids, dtype=np.int32),
        offsets=np.asarray(offsets, dtype=np.int64),
        review_ids=np.asarray(review_ids, dtype=np.int64),
    )


def make_batches(
    lengths: np.ndarray,
    max_tokens: int,
    seed: int | None = None,
    pool_batches: int = POOL_BATCHES,
    fixed_widths: bool = True,
) -> list[np.ndarray]:
    """Group reviews into batches of at most max_tokens padded positions.

    Args:
        lengths: Tokens per review (without <bos>/<eos>); a review fills length + 1 positions.
        max_tokens: Most positions per batch: reviews in the batch times its longest review.
        seed: None for evaluation (sorted by length, fixed order). Otherwise the reviews
            are shuffled, cut into pools of about pool_batches batches, each pool is sorted
            by length and packed, and the batches are shuffled.
        pool_batches: Pool size in batches; larger pools waste less padding but mix less.
        fixed_widths: Pad batches to the fixed WIDTHS (see padded_width).

    Returns:
        One array of review positions per batch. Every review appears exactly once.
    """
    positions = np.asarray(lengths, dtype=np.int64) + 1
    if positions.size == 0:
        return []
    if padded_width(int(positions.max()), fixed_widths) > max_tokens:
        raise ValueError(f"max_tokens={max_tokens} is below the longest review ({positions.max()})")
    if seed is None:
        pools = [np.argsort(positions, kind="stable")]
        rng = None
    else:
        rng = np.random.default_rng(seed)
        order = rng.permutation(positions.size)
        per_batch = max(1, max_tokens // max(1, int(positions.mean())))
        pool_size = max(1, pool_batches * per_batch)
        pools = [order[start : start + pool_size] for start in range(0, order.size, pool_size)]
        pools = [pool[np.argsort(positions[pool], kind="stable")] for pool in pools]
    batches: list[np.ndarray] = []
    for pool in pools:
        current: list[int] = []
        longest = 0
        for i in pool.tolist():
            size = int(positions[i])
            width = padded_width(max(longest, size), fixed_widths)
            if current and width * (len(current) + 1) > max_tokens:
                batches.append(np.asarray(current, dtype=np.int64))
                current, longest = [], 0
            current.append(i)
            longest = max(longest, size)
        if current:
            batches.append(np.asarray(current, dtype=np.int64))
    if rng is not None:
        batches = [batches[i] for i in rng.permutation(len(batches))]
    return batches


def collate(
    reviews: EncodedReviews,
    index: np.ndarray,
    device: torch.device | str | None = None,
    fixed_widths: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Padded inputs and targets for the reviews at these positions.

    Row r holds <bos> w1 ... wn as inputs and w1 ... wn <eos> as targets, then
    <pad> up to the batch width (padded_width of the longest review). Padded
    targets are never scored.
    """
    lengths = reviews.lengths[index]
    width = padded_width(int(lengths.max()) + 1, fixed_widths)
    inputs = np.full((index.size, width), PAD_ID, dtype=np.int64)
    targets = np.full((index.size, width), PAD_ID, dtype=np.int64)
    inputs[:, 0] = BOS_ID
    for row, (i, n) in enumerate(zip(index.tolist(), lengths.tolist(), strict=True)):
        tokens = reviews.ids[reviews.offsets[i] : reviews.offsets[i] + n]
        inputs[row, 1 : n + 1] = tokens
        targets[row, :n] = tokens
        targets[row, n] = EOS_ID
    return torch.from_numpy(inputs).to(device), torch.from_numpy(targets).to(device)
