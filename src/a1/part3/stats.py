"""How sure we are about Part III results (D3.3).

- Error bars: a 95% percentile bootstrap over the test reviews. One set of
  resamples is shared by every representation, so differences are paired.
  Accuracy inside a length group gets a Wilson interval instead (the groups
  are subsets; their size is reported with them).
- A check on those error bars: reviews of one product are not independent
  (their sentiment correlates, about 0.17, D0.10), so the same intervals are
  also computed by resampling whole products (a cluster bootstrap); if they
  are much wider, the review-level intervals and McNemar p-values are too
  optimistic and the report says so.
- Is a gap real: McNemar's exact test on the same test reviews, counting the
  reviews only one of the two classifiers gets right (Dietterich 1998). The
  comparisons fixed in advance form one family, corrected with Holm's
  step-down method (Holm 1979).
- Does better next-token prediction give better features: Kendall's tau
  between -J and accuracy over the models. Without ties, (1 + tau) / 2 is the
  share of model pairs put in the same order by both.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats


def bootstrap_accuracies(
    correct: np.ndarray, n_boot: int, seed: int, groups: np.ndarray | None = None
) -> np.ndarray:
    """Accuracy of each row of `correct` (representations x test reviews) on n_boot resamples.

    Each resample draws the test reviews with replacement (or, with `groups`,
    whole groups such as products, each bringing all its reviews); the same
    draws serve every row, so the rows' differences are paired. Returns
    (n_boot, rows).
    """
    correct = np.asarray(correct, dtype=np.float64)
    if correct.ndim == 1:
        correct = correct[None, :]
    rng = np.random.default_rng(seed)
    n = correct.shape[1]
    if groups is None:
        members = np.arange(n)
        count = n
    else:
        _, members = np.unique(np.asarray(groups), return_inverse=True)
        count = int(members.max()) + 1
        if members.size != n:
            raise ValueError("One group per review is needed")
    samples = np.empty((n_boot, correct.shape[0]))
    for b in range(n_boot):
        drawn = np.bincount(rng.integers(0, count, count), minlength=count)
        weights = drawn[members].astype(np.float64)
        samples[b] = correct @ weights / weights.sum()
    return samples


def interval(samples: np.ndarray, level: float = 0.95) -> list[float]:
    """Percentile interval of bootstrap samples."""
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(np.asarray(samples, dtype=np.float64), [tail, 1.0 - tail])
    return [float(low), float(high)]


def wilson(successes: float, total: int, level: float = 0.95) -> list[float] | None:
    """Wilson score interval for a proportion."""
    if total == 0:
        return None
    z = float(stats.norm.ppf(0.5 + level / 2.0))
    share = successes / total
    centre = (share + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(share * (1 - share) / total + z * z / (4 * total * total))
    half /= 1 + z * z / total
    return [centre - half, centre + half]


def mcnemar(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    """Exact McNemar test: do two classifiers' errors on the same reviews differ?

    first, second: booleans, True where that classifier is right. Only the
    reviews exactly one of them gets right count; under no difference each is
    equally likely to be either, so their split follows Binomial(n, 1/2).
    """
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    if first.shape != second.shape:
        raise ValueError("Both classifiers must be scored on the same reviews")
    first_only = int((first & ~second).sum())
    second_only = int((~first & second).sum())
    total = first_only + second_only
    p = 1.0 if total == 0 else float(stats.binomtest(min(first_only, second_only), total).pvalue)
    return {"first_only": first_only, "second_only": second_only, "p": p}


def holm(pvalues: list[float]) -> list[float]:
    """Holm's step-down adjustment: the k-th smallest p is multiplied by (m - k + 1)."""
    p = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (p.size - rank) * p[index]))
        adjusted[index] = running
    return adjusted.tolist()


def pair_agreement(j: np.ndarray, accuracy: np.ndarray) -> dict[str, Any]:
    """How often the model that predicts the next word better (lower J) also classifies better."""
    j = np.asarray(j, dtype=np.float64)
    accuracy = np.asarray(accuracy, dtype=np.float64)
    concordant = discordant = ties = 0
    for i in range(j.size):
        for k in range(i + 1, j.size):
            sign = np.sign(j[k] - j[i]) * np.sign(accuracy[i] - accuracy[k])
            if sign > 0:
                concordant += 1
            elif sign < 0:
                discordant += 1
            else:
                ties += 1
    result = stats.kendalltau(-j, accuracy)
    decided = concordant + discordant
    return {
        "models": int(j.size),
        "pairs": int(j.size * (j.size - 1) // 2),
        "same_order": concordant,
        "opposite_order": discordant,
        "ties": ties,
        "share_same_order": concordant / decided if decided else None,
        "tau": None if math.isnan(result.statistic) else float(result.statistic),
        "p": None if math.isnan(result.pvalue) else float(result.pvalue),
    }


def agreement_interval(j: np.ndarray, samples: np.ndarray, level: float = 0.95) -> list[float]:
    """Interval for tau over bootstrap accuracy samples (n_boot x models)."""
    taus = [stats.kendalltau(-np.asarray(j), row).statistic for row in samples]
    taus = np.asarray([t for t in taus if not math.isnan(t)])
    return interval(taus, level) if taus.size else [math.nan, math.nan]


def class_scores(labels: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Accuracy, macro-F1 and each class's accuracy (recall)."""
    labels = np.asarray(labels)
    predicted = np.asarray(predicted)
    scores: dict[str, float] = {"accuracy": float((labels == predicted).mean())}
    f1s = []
    for label, name in ((0, "negative"), (1, "positive")):
        truth = labels == label
        guess = predicted == label
        hits = float((truth & guess).sum())
        recall = hits / truth.sum() if truth.sum() else 0.0
        precision = hits / guess.sum() if guess.sum() else 0.0
        f1s.append(
            0.0 if recall + precision == 0 else 2 * recall * precision / (recall + precision)
        )
        scores[f"accuracy_{name}"] = recall
    scores["macro_f1"] = float(np.mean(f1s))
    return scores
