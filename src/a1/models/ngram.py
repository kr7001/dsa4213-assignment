"""Count-based n-gram language models (Lecture 1), on the same tokens as the neural models.

Both models interpolate every history length up to the order n:

- AddK, the course's model (Lecture1_Demo, sections 5 to 7; Lecture3_Demo):
  P(w | h) = sum_m lambda_m (c(h_m, w) + k) / (c(h_m) + k |V|), where h_m is the
  last m - 1 tokens of the history. The weights lambda are fitted by EM on
  validation data and k is picked from a grid, also on validation data.
- KneserNey, interpolated modified Kneser-Ney (Chen and Goodman, 1998): each
  order subtracts a discount D1, D2 or D3+ from counts of 1, 2 or 3+, gives the
  freed mass to the next shorter history, and lower orders count how many
  different words come before an n-gram (continuation counts) instead of how
  often it occurs. As in KenLM (Heafield et al., 2013, section 3.2), n-grams that
  begin with <bos> keep their raw counts, since nothing can come before <bos>.
  The discounts come from the counts themselves, so nothing is tuned.

Scoring follows D0.5: each review is scored on its own, histories never reach
back past <bos>, and the targets are every token after <bos>, including <eos>.
Near the start of a review, where a full history does not exist, a model uses
the longest history that does.

Counting uses integer keys instead of dictionaries. A window of L tokens gets
the key rank(first L - 1 tokens) * |V| + last token, where rank is the position
of the shorter window in the sorted list of distinct windows of that length.
Keys stay far below 2^63 for any order used here, so numpy sorting and binary
search do all the work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from a1.data.sequences import EncodedReviews
from a1.data.vocab import BOS_ID, EOS_ID

ADD_K_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0)
K_FLOOR, K_CEILING = 1e-8, 100.0  # how far the grid may be extended past its edges
EM_ITERATIONS = 200


def next_k(results: dict[float, float]) -> float | None:
    """The next k to try for add-k, given validation J per k tried so far.

    First the base grid. Then, while the best k is the smallest (or largest)
    value tried, one more step 10x further that way, so the chosen k is never
    left at the edge of the values tried: the same idea as the learning-rate
    rule (src/a1/training/tuning.py), extended until the best is inside.
    """
    for k in ADD_K_GRID:
        if k not in results:
            return k
    best = min(sorted(results), key=lambda k: results[k])
    smallest, largest = min(results), max(results)
    if best == smallest and smallest / 10 >= K_FLOOR * 0.999:
        return float(f"{smallest / 10:.0e}")
    if best == largest and largest * 10 <= K_CEILING * 1.001:
        return float(f"{largest * 10:.0e}")
    return None


def flat_sequences(reviews: EncodedReviews) -> tuple[np.ndarray, np.ndarray]:
    """Every review as <bos> w1 ... wn <eos>, back to back, and each token's offset from <bos>."""
    lengths = reviews.lengths
    sizes = lengths + 2
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    tokens = np.empty(int(sizes.sum()), dtype=np.int64)
    body = np.ones(tokens.size, dtype=bool)
    body[starts] = False
    body[starts + lengths + 1] = False
    tokens[starts] = BOS_ID
    tokens[starts + lengths + 1] = EOS_ID
    tokens[body] = reviews.ids
    offsets = np.arange(tokens.size, dtype=np.int64) - np.repeat(starts, sizes)
    return tokens, offsets


def _lookup(keys: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """Index of each wanted key in the sorted keys, or -1 where it is missing."""
    if keys.size == 0:
        return np.full(wanted.shape, -1, dtype=np.int64)
    index = np.searchsorted(keys, wanted)
    index = np.minimum(index, keys.size - 1)
    return np.where(keys[index] == wanted, index, -1)


@dataclass
class NgramCounts:
    """Counts of every window of 1 to max_order tokens in the training reviews.

    Attributes (lists indexed by window length L, entry 0 unused):
        keys: Sorted distinct window keys.
        counts: How often each window occurs.
        from_bos: Whether the window starts with <bos> (it can only open a review).
        continuation: For L < max_order, how many different tokens come right
            before the window (Kneser-Ney continuation counts).
    """

    vocab_size: int
    max_order: int
    keys: list = field(default_factory=lambda: [None])
    counts: list = field(default_factory=lambda: [None])
    from_bos: list = field(default_factory=lambda: [None])
    continuation: list = field(default_factory=lambda: [None])
    targets: int = 0

    @classmethod
    def build(cls, reviews: EncodedReviews, vocab_size: int, max_order: int) -> NgramCounts:
        table = cls(vocab_size, max_order)
        tokens, offsets = flat_sequences(reviews)
        table.targets = int((offsets >= 1).sum())
        previous_rank = np.zeros(0, dtype=np.int64)
        for length in range(1, max_order + 1):
            valid = np.nonzero(offsets >= length - 1)[0]
            if length == 1:
                windows = tokens[valid]
            else:  # every valid window's prefix is a valid shorter window ending one earlier
                windows = previous_rank[valid - 1] * vocab_size + tokens[valid]
            keys, first, counts = np.unique(windows, return_index=True, return_counts=True)
            rank = np.full(tokens.size, -1, dtype=np.int64)
            rank[valid] = np.searchsorted(keys, windows)
            from_bos = np.zeros(keys.size, dtype=bool)
            from_bos[rank[offsets == length - 1]] = True
            if length >= 2:  # the suffix of each distinct window is a window one token shorter
                suffix_rank = previous_rank[valid[first]]
                table.continuation.append(
                    np.bincount(suffix_rank, minlength=table.keys[length - 1].size)
                )
            table.keys.append(keys)
            table.counts.append(counts)
            table.from_bos.append(from_bos)
            previous_rank = rank
        table.continuation.append(None)  # the highest order has no longer windows
        return table

    def stored_entries(self, order: int) -> int:
        """Distinct n-grams kept for a model of this order: the size of the count model."""
        return int(sum(self.keys[m].size for m in range(1, order + 1)))

    def nbytes(self, order: int) -> int:
        """Bytes of the count arrays a model of this order reads (keys, counts, flags)."""
        total = 0
        for m in range(1, order + 1):
            total += self.keys[m].nbytes + self.counts[m].nbytes + self.from_bos[m].nbytes
            if m < order and self.continuation[m] is not None:
                total += self.continuation[m].nbytes
        return int(total)

    def history_ranks(self, tokens: np.ndarray, offsets: np.ndarray, order: int) -> list:
        """For windows of 1 to order-1 tokens ending at each position, their rank (or -1).

        A window is missing when it never occurred in training or when it would
        reach back past <bos>.
        """
        ranks = [None]
        current = _lookup(self.keys[1], tokens)
        ranks.append(current)
        for length in range(2, order):
            following = np.full(tokens.size, -1, dtype=np.int64)
            position = np.nonzero(offsets >= length - 1)[0]
            prefix = current[position - 1]
            known = prefix >= 0
            position = position[known]
            following[position] = _lookup(
                self.keys[length], prefix[known] * self.vocab_size + tokens[position]
            )
            ranks.append(following)
            current = following
        return ranks


DISCOUNT_FALLBACK = (0.5, 1.0, 1.5)  # KenLM's default --discount_fallback values


def discounts(counts: np.ndarray) -> np.ndarray:
    """Modified Kneser-Ney discounts [0, D1, D2, D3+] from the counts' count-of-counts.

    Chen and Goodman (1998): Y = n1 / (n1 + 2 n2), D1 = 1 - 2Y n2/n1,
    D2 = 2 - 3Y n3/n2, D3+ = 3 - 4Y n4/n3, where nr is how many n-grams have
    count r. A discount must lie strictly between 0 and its count, or unseen
    words could get zero probability; when the formula leaves that range (only
    on tiny or artificial data), the discount falls back to 0.5, 1 or 1.5, as
    KenLM's --discount_fallback does.
    """
    n = [int((counts == r).sum()) for r in range(5)]
    y = n[1] / (n[1] + 2 * n[2]) if n[1] + 2 * n[2] > 0 else 0.0
    values = [0.0]
    for r in (1, 2, 3):
        value = r - (r + 1) * y * n[r + 1] / n[r] if n[r] > 0 else -1.0
        values.append(value if 0.0 < value < r else DISCOUNT_FALLBACK[r - 1])
    return np.array(values)


def _discount_of(counts: np.ndarray, d: np.ndarray) -> np.ndarray:
    return d[np.minimum(counts, 3).astype(np.int64)]


class KneserNey:
    """Interpolated modified Kneser-Ney (Chen and Goodman, 1998) of a given order."""

    def __init__(self, table: NgramCounts, order: int) -> None:
        if not 1 <= order <= table.max_order:
            raise ValueError(f"order must be between 1 and {table.max_order}")
        self.table, self.order = table, order
        vocab = table.vocab_size
        self.levels: list[dict] = []
        for m in range(1, order + 1):
            keys = table.keys[m]
            raw = table.counts[m]
            if m == order:
                c = raw.astype(np.float64)
            else:  # continuation counts, but raw counts for n-grams that open a review
                c = np.where(table.from_bos[m], raw, table.continuation[m]).astype(np.float64)
            if m == 1:
                c[keys == BOS_ID] = 0.0  # <bos> is never predicted
            d = discounts(c)
            level: dict = {"keys": keys, "c": c, "d": d}
            if m == 1:
                total = c.sum()
                kinds = [(c == 1).sum(), (c == 2).sum(), (c >= 3).sum()]
                gamma = (d[1] * kinds[0] + d[2] * kinds[1] + d[3] * kinds[2]) / total
                p = np.full(vocab, gamma / vocab)  # the freed mass is spread uniformly
                p[keys] += np.maximum(c - _discount_of(c, d), 0.0) / total
                level["p"] = p
            else:
                history = keys // vocab
                size = table.keys[m - 1].size
                total = np.bincount(history, weights=c, minlength=size)
                freed = np.bincount(history, weights=_discount_of(c, d) * (c > 0), minlength=size)
                level["total"] = total
                level["gamma"] = np.divide(freed, total, out=np.zeros(size), where=total > 0)
            self.levels.append(level)

    def _raise(self, m: int, p_lower: np.ndarray, history: np.ndarray, words: np.ndarray):
        """P_m from P_{m-1} for each (history rank, word); unknown histories keep P_{m-1}."""
        level = self.levels[m - 1]
        p = p_lower.copy()
        known = history >= 0
        h = history[known]
        total = level["total"][h]
        index = _lookup(level["keys"], h * self.table.vocab_size + words[known])
        c = np.where(index >= 0, level["c"][np.maximum(index, 0)], 0.0)
        kept = np.maximum(c - _discount_of(c, level["d"]), 0.0)
        seen = total > 0
        update = np.where(
            seen,
            kept / np.where(seen, total, 1.0) + level["gamma"][h] * p_lower[known],
            p_lower[known],
        )
        p[known] = update
        return p

    def log_probs(self, reviews: EncodedReviews) -> np.ndarray:
        """Natural-log probability of every target, in review order."""
        tokens, offsets = flat_sequences(reviews)
        target = np.nonzero(offsets >= 1)[0]
        words = tokens[target]
        p = self.levels[0]["p"][words]
        ranks = self.table.history_ranks(tokens, offsets, self.order)
        for m in range(2, self.order + 1):
            p = self._raise(m, p, ranks[m - 1][target - 1], words)
        return np.log(p)

    def distribution(self, history: list[int]) -> np.ndarray:
        """P(w | history) for every w in the vocabulary; history starts with <bos>."""
        vocab = self.table.vocab_size
        words = np.arange(vocab, dtype=np.int64)
        p = self.levels[0]["p"].copy()
        ranks = _history_ranks_of(self.table, history, self.order)
        for m in range(2, self.order + 1):
            p = self._raise(m, p, np.full(vocab, ranks[m - 1]), words)
        return p


def _history_ranks_of(table: NgramCounts, history: list[int], order: int) -> list[int]:
    """Ranks of the last 1 to order-1 tokens of one history (-1 when unseen or too long)."""
    tokens = np.asarray(history, dtype=np.int64)
    offsets = np.arange(tokens.size, dtype=np.int64)
    ranks = table.history_ranks(tokens, offsets, max(order, 2))
    return [-1] + [int(ranks[length][-1]) for length in range(1, order)]


class AddK:
    """The course's interpolated add-k model, with lambda fitted by EM on validation data."""

    def __init__(self, table: NgramCounts, order: int, k: float, weights=None) -> None:
        self.table, self.order, self.k = table, order, k
        vocab = table.vocab_size
        self.levels: list[dict] = []
        for m in range(1, order + 1):
            keys = table.keys[m]
            c = table.counts[m].astype(np.float64)
            if m == 1:
                c[keys == BOS_ID] = 0.0
                total = c.sum()
                p = np.full(vocab, k / (total + k * vocab))
                p[keys] = (c + k) / (total + k * vocab)
                self.levels.append({"p": p})
            else:
                history = keys // vocab
                total = np.bincount(history, weights=c, minlength=table.keys[m - 1].size)
                self.levels.append({"keys": keys, "c": c, "total": total})
        self.weights = np.full(order, 1.0 / order) if weights is None else np.asarray(weights)

    def _component(self, m: int, p_lower: np.ndarray, history: np.ndarray, words: np.ndarray):
        """Add-k estimate with the last m-1 tokens; histories that cannot exist keep P_{m-1}."""
        level = self.levels[m - 1]
        vocab = self.table.vocab_size
        p = p_lower.copy()
        known = history >= 0
        h = history[known]
        index = _lookup(level["keys"], h * vocab + words[known])
        c = np.where(index >= 0, level["c"][np.maximum(index, 0)], 0.0)
        p[known] = (c + self.k) / (level["total"][h] + self.k * vocab)
        return p

    def components(self, reviews: EncodedReviews) -> np.ndarray:
        """One column per order: each target's add-k probability with that history length.

        A history seen in no training review gives 1/|V|. A history that would
        reach back past <bos> uses the next shorter order instead.
        """
        tokens, offsets = flat_sequences(reviews)
        target = np.nonzero(offsets >= 1)[0]
        words = tokens[target]
        ranks = self.table.history_ranks(tokens, offsets, self.order)
        columns = [self.levels[0]["p"][words]]
        vocab = self.table.vocab_size
        for m in range(2, self.order + 1):
            history = ranks[m - 1][target - 1]
            fits = offsets[target] >= m - 1  # the full history lies inside the review
            unseen = fits & (history < 0)
            column = self._component(m, columns[-1], history, words)
            column[unseen] = 1.0 / vocab  # c(h) = 0: add-k gives every word k / (k |V|)
            columns.append(column)
        return np.stack(columns, axis=1)

    def fit_weights(self, reviews: EncodedReviews, iterations: int = EM_ITERATIONS) -> None:
        """EM for lambda: maximise the validation log-likelihood (Lecture1_Demo, section 7)."""
        parts = self.components(reviews)
        weights = np.full(self.order, 1.0 / self.order)
        for _ in range(iterations):
            share = parts * weights
            share /= share.sum(axis=1, keepdims=True)
            weights = share.mean(axis=0)
        self.weights = weights

    def log_probs(self, reviews: EncodedReviews) -> np.ndarray:
        return np.log(self.components(reviews) @ self.weights)

    def distribution(self, history: list[int]) -> np.ndarray:
        vocab = self.table.vocab_size
        words = np.arange(vocab, dtype=np.int64)
        ranks = _history_ranks_of(self.table, history, self.order)
        columns = [self.levels[0]["p"].copy()]
        for m in range(2, self.order + 1):
            fits = len(history) >= m - 1
            if fits and ranks[m - 1] < 0:
                columns.append(np.full(vocab, 1.0 / vocab))
            else:
                columns.append(self._component(m, columns[-1], np.full(vocab, ranks[m - 1]), words))
        return np.stack(columns, axis=1) @ self.weights


def unseen_shares(table: NgramCounts, reviews: EncodedReviews, order: int) -> dict[int, float]:
    """For each n up to order: the share of targets whose full n-gram never occurred in training.

    Only targets far enough from <bos> to have a full history of n - 1 tokens count.
    """
    tokens, offsets = flat_sequences(reviews)
    target = np.nonzero(offsets >= 1)[0]
    words = tokens[target]
    ranks = table.history_ranks(tokens, offsets, order)
    shares = {1: float((_lookup(table.keys[1], words) < 0).mean())}
    for n in range(2, order + 1):
        fits = offsets[target] >= n - 1
        history = ranks[n - 1][target - 1][fits]
        found = np.zeros(history.size, dtype=bool)
        known = history >= 0
        found[known] = (
            _lookup(table.keys[n], history[known] * table.vocab_size + words[fits][known]) >= 0
        )
        shares[n] = float(1.0 - found.mean()) if found.size else float("nan")
    return shares
