"""The stopping rule (D0.6, locked 12 Oct).

Validation J is checked at fixed points (4 per epoch). Any improvement counts
as a new best; training stops after `patience` checks in a row with no new best,
and the best checkpoint is kept. This is the early-stopping algorithm of
Goodfellow, Bengio and Courville (2016, section 7.8, algorithm 7.1). Waiting a
few checks matters because real validation curves "almost always have more
than one local minimum" (Prechelt, 1998).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

PATIENCE = 3


@dataclass
class EarlyStopping:
    patience: int = PATIENCE
    best: float = math.inf
    best_step: int = -1
    checks_without_best: int = 0

    def update(self, value: float, step: int) -> bool:
        """Record a validation J; return True if it is a new best."""
        if value < self.best:
            self.best, self.best_step, self.checks_without_best = value, step, 0
            return True
        self.checks_without_best += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.checks_without_best >= self.patience
