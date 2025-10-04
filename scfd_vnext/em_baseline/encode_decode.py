from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np


def encode_input(state: np.ndarray, inputs: Sequence[float], reward: float | None = None) -> np.ndarray:
    encoded = state.copy()
    n = min(len(inputs), state.shape[0])
    encoded[:n, 0] = np.asarray(inputs[:n])
    if reward is not None:
        encoded[-n:, -1] = reward
    return encoded


def decode_output(state: np.ndarray, n_symbols: int = 10) -> Dict[str, np.ndarray]:
    right_edge = state[:, -1]
    upper_band = state[0:n_symbols]
    lower_band = state[-n_symbols:]
    accel = np.tanh(right_edge.mean())
    steering = np.tanh(upper_band.mean() - lower_band.mean())
    symbol_bins = np.array_split(right_edge, n_symbols)
    symbols = [int((bin.mean() > 0).item()) for bin in symbol_bins]
    return {
        "actions": np.array([accel, steering]),
        "symbols": np.asarray(symbols, dtype=int),
    }
