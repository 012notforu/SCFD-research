from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class HaltingController:
    threshold: float
    max_steps: int
    stochastic_p: float = 0.0
    rng: Optional[np.random.Generator] = None

    def __post_init__(self) -> None:
        self.rng = self.rng or np.random.default_rng()

    def should_halt(self, state: np.ndarray, step: int) -> bool:
        if step >= self.max_steps:
            return True
        halt_value = float(state[state.shape[0] // 2, state.shape[1] // 2])
        if halt_value > self.threshold:
            return True
        if self.stochastic_p > 0.0 and self.rng.random() < self.stochastic_p:
            return True
        return False
