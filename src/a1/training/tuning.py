"""The tuning rule (D0.6, locked 12 Oct).

Every neural model tries the same three peak learning rates, 3x apart on a log
scale and centred on Adam's suggested 0.001 (Kingma and Ba, 2015), and keeps
the one with the lowest validation J. The learning rate is the one setting
worth tuning when only one can be (Goodfellow, Bengio and Courville, 2016,
section 11.4.1), and which model looks best depends on the tuning budget
(Dodge et al., 2019), so the budget is the same for every model. If the best
value is the smallest or the largest, the model tries one more step that way,
once; the same rule applies to every model, and the report states each model's
tries.
"""

from __future__ import annotations

import math

LEARNING_RATES = (3e-4, 1e-3, 3e-3)
EXTRA_STEP = {3e-4: 1e-4, 3e-3: 1e-2}  # one more step beyond each edge of the grid


def next_learning_rate(results: dict[float, float]) -> float | None:
    """The next learning rate to try, given validation J per rate tried so far.

    A diverged or failed try should be recorded as math.inf.
    """
    for rate in LEARNING_RATES:
        if rate not in results:
            return rate
    best = best_learning_rate({rate: results[rate] for rate in LEARNING_RATES})
    extra = EXTRA_STEP.get(best)
    if extra is not None and extra not in results:
        return extra
    return None


def best_learning_rate(results: dict[float, float]) -> float:
    """The rate with the lowest validation J (the smaller rate wins a tie)."""
    finite = {rate: j for rate, j in results.items() if math.isfinite(j)}
    if not finite:
        raise ValueError("Every learning rate diverged")
    return min(sorted(finite), key=lambda rate: finite[rate])
