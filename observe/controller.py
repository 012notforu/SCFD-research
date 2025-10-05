from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


ALLOWED_NUDGES = {"T", "alpha", "gamma"}


@dataclass
class ControllerDecision:
    nudges: Dict[str, float]
    safe_mode: bool = False


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

    def _clamp(self, value: float) -> float:
        return float(np.clip(value, -self.max_step, self.max_step))

    def step(self, metrics: Dict[str, float]) -> ControllerDecision:
        spectrum = self._update_ema("spectrum", metrics.get("spectrum_width", 0.0))
        perplexity = self._update_ema("perplexity", metrics.get("perplexity", 0.0))
        horizon = self._update_ema("horizon", metrics.get("horizon", 0.0))
        drift = metrics.get("energy_drift", 0.0)
        adjustments = {
            "T": self._clamp(-self.gains["spectrum"] * (spectrum - self.surprise_threshold)),
            "alpha": self._clamp(-self.gains["perplexity"] * (perplexity - self.surprise_threshold)),
            "gamma": self._clamp(self.gains["horizon"] * (self.surprise_threshold - horizon)),
        }
        self._validate_nudges(adjustments)
        safe_mode = False
        if abs(drift) > 0.01:
            safe_mode = True
            adjustments = {key: 0.0 for key in ALLOWED_NUDGES}
        return ControllerDecision(nudges=adjustments, safe_mode=safe_mode)

    def _validate_nudges(self, nudges: Dict[str, float]) -> None:
        unknown = set(nudges) - ALLOWED_NUDGES
        if unknown:
            raise ValueError(f"Unsupported nudge parameters: {sorted(unknown)}")
