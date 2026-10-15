"""One classifier for every representation (D3.3).

Logistic regression with an L2 penalty: a single straight-line boundary in the
feature space, so it shows what the features already contain rather than
learning the task itself (Hewitt and Liang 2019 found linear probes
"considerably more selective" than multi-layer perceptrons). Its strength C is
chosen by 5-fold cross-validation on part3_train only:

- The folds split by product, as the test pile does, because two reviews of a
  product tend to share sentiment (correlation about 0.17, D0.10); they are
  stratified by label and are the same for every representation, so every
  comparison is paired.
- Each fit prepares its features on its fitting part only: dense features are
  standardised with that part's mean and standard deviation; word counts keep
  the n-grams found in at least `min_documents` fitting reviews and become
  tf-idf weights computed there. The held-out part and the test pile are
  transformed with those numbers.
- C runs over powers of ten. If the best average accuracy is at an end of the
  range, the range grows by one power on that side (the rule used for the
  learning rates, D0.6), within fixed limits. Ties go to the smaller C (the
  stronger penalty), and a flat stretch of equal scores never pulls the range
  (see next_power).

The test pile is used once, after C is fixed, by a model refit on all of
part3_train.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

Matrix = np.ndarray | sparse.csr_matrix
Transform = Callable[[Matrix], Matrix]


def make_folds(
    labels: np.ndarray, groups: np.ndarray, n_folds: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Folds stratified by label that never split a product (StratifiedGroupKFold)."""
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    placeholder = np.zeros((labels.size, 1))
    return [(fit, held) for fit, held in splitter.split(placeholder, labels, groups)]


class DenseFeatures:
    """A dense feature matrix for the training and test reviews; standardised per fit."""

    def __init__(self, train: np.ndarray, test: np.ndarray) -> None:
        if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
            raise ValueError(f"Bad feature shapes {train.shape} and {test.shape}")
        self.train = train
        self.test = test

    @property
    def width(self) -> int:
        return int(self.train.shape[1])

    def rows(self, rows: np.ndarray) -> np.ndarray:
        return self.train[rows]

    def fit(self, rows: np.ndarray) -> Transform:
        """Mean and standard deviation of the fitting rows (in float64), applied in float32.

        float32 halves the memory the solver reads at every step; the
        solver's iterations and answers match float64 in tests (to its 1e-4
        tolerance), and the largest features are 1,120 numbers wide.
        """
        fit = np.asarray(self.train[rows], dtype=np.float64)
        mean = fit.mean(axis=0)
        scale = fit.std(axis=0)
        scale[scale == 0] = 1.0

        def transform(matrix: Matrix) -> np.ndarray:
            return ((np.asarray(matrix, dtype=np.float64) - mean) / scale).astype(np.float32)

        return transform


class CountFeatures:
    """N-gram counts for the training and test reviews; filtered and tf-idf weighted per fit."""

    def __init__(
        self,
        train: sparse.csr_matrix,
        test: sparse.csr_matrix,
        min_documents: int = 2,
        sublinear: bool = True,
    ) -> None:
        if train.shape[1] != test.shape[1]:
            raise ValueError("Training and test counts must share their columns")
        if min_documents < 1:
            raise ValueError(
                "min_documents must be at least 1, so n-grams absent from the "
                "fitting rows are always dropped"
            )
        self.train = sparse.csr_matrix(train)
        self.test = sparse.csr_matrix(test)
        self.min_documents = min_documents
        self.sublinear = sublinear
        self.last_width = 0

    @property
    def width(self) -> int:
        return self.last_width

    def rows(self, rows: np.ndarray) -> sparse.csr_matrix:
        return self.train[rows]

    def fit(self, rows: np.ndarray) -> Transform:
        fit = self.train[rows]
        documents = np.asarray((fit > 0).sum(axis=0)).ravel()
        kept = np.flatnonzero(documents >= self.min_documents)
        weights = TfidfTransformer(sublinear_tf=self.sublinear).fit(fit[:, kept])
        self.last_width = int(kept.size)

        def transform(matrix: Matrix) -> sparse.csr_matrix:
            return weights.transform(sparse.csr_matrix(matrix)[:, kept])

        return transform


Features = DenseFeatures | CountFeatures


@dataclass
class ClassifierSettings:
    low: int = -4
    high: int = 2
    lowest: int = -8
    highest: int = 6
    max_iter: int = 5000


def fit_logistic(
    features: Matrix, labels: np.ndarray, power: int, max_iter: int
) -> tuple[LogisticRegression, bool]:
    """L2 logistic regression with C = 10**power, and whether the solver converged.

    A fit that stops at max_iter is kept but reported, so one slow candidate
    far out in the C range cannot stop a long run; the results flag it.
    """
    model = LogisticRegression(C=10.0**power, solver="lbfgs", max_iter=max_iter)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(features, labels)
    converged = True
    for warning in caught:
        if issubclass(warning.category, ConvergenceWarning):
            converged = False
        else:  # anything else (overflow, line search) is shown as usual
            warnings.showwarning(
                warning.message, warning.category, warning.filename, warning.lineno
            )
    return model, converged


def cross_validate(
    features: Features,
    labels: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    powers: list[int],
    max_iter: int,
) -> tuple[dict[int, list[float]], dict[int, int]]:
    """Held-out accuracy of every fold for each power of ten (C = 10**power).

    Also returns, for each power, how many of its fold fits did not converge.
    """
    scores: dict[int, list[float]] = {power: [] for power in powers}
    unconverged = dict.fromkeys(powers, 0)
    for fit_rows, held_rows in folds:
        transform = features.fit(fit_rows)
        fit_matrix = transform(features.rows(fit_rows))
        held_matrix = transform(features.rows(held_rows))
        for power in powers:
            model, converged = fit_logistic(fit_matrix, labels[fit_rows], power, max_iter)
            accuracy = (model.predict(held_matrix) == labels[held_rows]).mean()
            scores[power].append(float(accuracy))
            unconverged[power] += int(not converged)
    return scores, unconverged


def best_power(means: dict[int, float]) -> int:
    """The smallest power of ten with the highest mean accuracy (ties go to the smaller C)."""
    top = max(means.values())
    return min(power for power, value in means.items() if value == top)


def next_power(means: dict[int, float], settings: ClassifierSettings) -> int | None:
    """The power of ten to try next, or None once the best lies inside the range.

    The range grows upwards when its largest power is the best (it then beats
    every smaller one, as ties go to the smaller C), and downwards when its
    smallest power beats every other one. A flat stretch of equal scores, such
    as very small C values that all predict one class, never pulls the range.
    """
    best = best_power(means)
    low, high = min(means), max(means)
    if best == high and high < settings.highest:
        return high + 1
    unique = all(value < means[low] for power, value in means.items() if power != low)
    if best == low and low > settings.lowest and unique:
        return low - 1
    return None


def choose_power(
    features: Features,
    labels: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    settings: ClassifierSettings,
) -> dict[str, Any]:
    """Cross-validate C over powers of ten, widening the range while the best is at an end."""
    powers = list(range(settings.low, settings.high + 1))
    scores, unconverged = cross_validate(features, labels, folds, powers, settings.max_iter)
    extended = 0
    while True:
        means = {power: float(np.mean(values)) for power, values in scores.items()}
        new = next_power(means, settings)
        if new is None:
            break
        more, more_unconverged = cross_validate(features, labels, folds, [new], settings.max_iter)
        scores.update(more)
        unconverged.update(more_unconverged)
        extended += 1
    best = best_power(means)
    return {
        "power": best,
        "C": 10.0**best,
        "mean_accuracy": {str(power): means[power] for power in sorted(means)},
        "fold_accuracy": {str(power): scores[power] for power in sorted(scores)},
        "unconverged_fold_fits": {str(p): n for p, n in sorted(unconverged.items()) if n},
        "extended": extended,
        "at_limit": best in (settings.lowest, settings.highest),
    }


def fit_and_test(
    features: Features,
    labels: np.ndarray,
    test_labels: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    settings: ClassifierSettings,
) -> dict[str, Any]:
    """Choose C on the training folds, refit on every training review, score the test pile once."""
    start = time.perf_counter()
    choice = choose_power(features, labels, folds, settings)
    every_row = np.arange(labels.size)
    transform = features.fit(every_row)
    model, converged = fit_logistic(
        transform(features.rows(every_row)), labels, choice["power"], settings.max_iter
    )
    test_matrix = transform(features.test)
    probability = model.predict_proba(test_matrix)[:, 1]
    predicted = model.predict(test_matrix).astype(np.int8)
    return {
        **choice,
        "width": features.width,
        "final_fit_converged": converged,
        "iterations": int(np.max(model.n_iter_)),
        "seconds": time.perf_counter() - start,
        "probability": probability.astype(np.float32),
        "predicted": predicted,
        "correct": predicted == test_labels,
    }
