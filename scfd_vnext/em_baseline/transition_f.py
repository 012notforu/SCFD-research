from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import fftconvolve


@dataclass
class LeniaTransition:
    grid_shape: tuple[int, int]
    radius: int
    sigma: float
    activation: str = "tanh"
    stochastic_prob: float = 0.0
    rng: Optional[np.random.Generator] = None

    def __post_init__(self) -> None:
        self.kernel = self._build_kernel(self.radius, self.sigma)
        self.rng = self.rng or np.random.default_rng()

    def _build_kernel(self, radius: int, sigma: float) -> np.ndarray:
        size = 2 * radius + 1
        x = np.arange(size) - radius
        xx, yy = np.meshgrid(x, x, indexing="ij")
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel /= kernel.sum()
        return kernel

    def _activate(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return np.tanh(x)
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-x))
        if self.activation == "relu":
            return np.maximum(0.0, x)
        raise ValueError(f"Unknown activation: {self.activation}")

    def step(self, state: np.ndarray) -> np.ndarray:
        conv = fftconvolve(state, self.kernel, mode="same")
        updated = self._activate(conv)
        if self.stochastic_prob > 0.0:
            noise_mask = self.rng.random(self.grid_shape) < self.stochastic_prob
            updated = updated + noise_mask.astype(np.float32) * self.rng.normal(scale=0.05, size=self.grid_shape)
        return np.clip(updated, -1.0, 1.0)
