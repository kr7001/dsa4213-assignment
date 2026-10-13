"""The tuning rule (D0.6, locked 12 Oct, amended 13 Oct).

Every neural model starts with the same three peak learning rates, 3x apart on
a log scale and centred on Adam's suggested 0.001 (Kingma and Ba, 2015), and
keeps the one with the lowest validation J. The learning rate is the one
setting worth tuning when only one can be (Goodfellow, Bengio and Courville,
2016, section 11.4.1), and which model looks best depends on the tuning budget
(Dodge et al., 2019), so every model follows the same rule.

If the best rate is the smallest or the largest tried, the model tries the
next rate half a decade further that way, and keeps going while the best rate
stays at that edge: when the best value is on the border, "it is recommended in
that case to explore further, beyond that border" (Bengio, 2012, section
3.3.1). The search stops when the best rate is inside the rates tried, when a
try diverges (its J is inf, so the best rate is inside), or at the end of the
ladder (1e-5 to 0.1). The rule as locked on 12 Oct allowed one extra step only;
it was extended on 13 Oct, after the LSTM's best rate was again the largest
tried, and the extension applies to every model. The report states each
model's tries.
"""

from __future__ import annotations

import math

LEARNING_RATES = (3e-4, 1e-3, 3e-3)  # every model starts with these three
# Steps half a decade apart, like the starting grid; the search never leaves this ladder.
LADDER = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)


def next_learning_rate(
    results: dict[float, float], start: tuple[float, ...] = LEARNING_RATES
) -> float | None:
    """The next learning rate to try, given validation J per rate tried so far.

    A diverged or failed try should be recorded as math.inf. ``start`` is the
    first grid; Part I uses LEARNING_RATES for every model, and a config can
    give another grid of ladder steps (``tuning_start``), which then holds for
    every model in that config.
    """
    if any(rate not in LADDER for rate in start):
        raise ValueError(f"Every starting rate must be a ladder step: {LADDER}")
    for rate in start:
        if rate not in results:
            return rate
    best = best_learning_rate(results)
    tried = sorted(results)
    step = LADDER.index(best)
    if best == tried[0] and step > 0:
        return LADDER[step - 1]
    if best == tried[-1] and step < len(LADDER) - 1:
        return LADDER[step + 1]
    return None


def best_learning_rate(results: dict[float, float]) -> float:
    """The rate with the lowest validation J (the smaller rate wins a tie)."""
    finite = {rate: j for rate, j in results.items() if math.isfinite(j)}
    if not finite:
        raise ValueError("Every learning rate diverged")
    return min(sorted(finite), key=lambda rate: finite[rate])
