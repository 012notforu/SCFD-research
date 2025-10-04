from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from .encode_decode import decode_output, encode_input
from .halting import HaltingController
from .transition_f import LeniaTransition


@dataclass
class EMRunConfig:
    grid_shape: tuple[int, int]
    steps: int
    kernel_radius: int
    kernel_sigma: float
    activation: str
    halting: Dict[str, float]


def run_em_episode(
    config: EMRunConfig,
    inputs: Sequence[float],
    reward: Optional[float] = None,
    initial_state: Optional[np.ndarray] = None,
    stochastic_prob: float = 0.0,
) -> Dict[str, object]:
    state = initial_state.copy() if initial_state is not None else np.zeros(config.grid_shape)
    state = encode_input(state, inputs, reward)
    transition = LeniaTransition(
        grid_shape=config.grid_shape,
        radius=config.kernel_radius,
        sigma=config.kernel_sigma,
        activation=config.activation,
        stochastic_prob=stochastic_prob,
    )
    halting = HaltingController(
        threshold=config.halting.get("threshold", 0.9),
        max_steps=config.halting.get("max_steps", config.steps),
        stochastic_p=config.halting.get("stochastic_p", 0.0),
    )
    steps = 0
    for step in range(config.steps):
        state = transition.step(state)
        steps = step + 1
        if halting.should_halt(state, step):
            break
    decoded = decode_output(state)
    decoded["state"] = state
    decoded["steps"] = steps
    return decoded
