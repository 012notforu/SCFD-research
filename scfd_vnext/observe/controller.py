from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class GentleController:
    max_step: float = 0.1
    ema_tau: float = 100.0
    surprise_threshold: float = 2.0
    gains: Dict[str, float] = field(default_factory=lambda: {
        "spectrum": 0.05,
        "perplexity": 0.03,
        "horizon": 0.02,
    })

    def __post_init__(self) -> None:
        self._ema: Dict[str, float] = {}

    def _update_ema(self, key: str, value: float) -> float:
        prev = self._ema.get(key, value)
        decay = np.exp(-1.0 / max(self.ema_tau, 1.0))
        updated = decay * prev + (1.0 - decay) * value
        self._ema[key] = updated
        return updated

    def step(self, metrics: Dict[str, float]) -> Dict[str, float]:
        spectrum = self._update_ema("spectrum", metrics.get("spectrum_width", 0.0))
        perplexity = self._update_ema("perplexity", metrics.get("perplexity", 0.0))
        horizon = self._update_ema("horizon", metrics.get("horizon", 0.0))
        drift = metrics.get("energy_drift", 0.0)
        adjustments = {
            "T": self._clamp(-self.gains["spectrum"] * (spectrum - self.surprise_threshold)),
            "alpha": self._clamp(-self.gains["perplexity"] * (perplexity - self.surprise_threshold)),
            "gamma": self._clamp(self.gains["horizon"] * (self.surprise_threshold - horizon)),
        }
        if abs(drift) > 0.01:
            adjustments = {k: 0.0 for k in adjustments}
            adjustments["dt_scale"] = 0.9
        return adjustments

    def _clamp(self, value: float) -> float:
        return float(np.clip(value, -self.max_step, self.max_step))
