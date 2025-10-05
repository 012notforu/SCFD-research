from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np


@dataclass
class RandomStateOptimizer:
    grid_shape: tuple[int, int]
    population: int
    elite: int
    mutation_scale: float
    generations: int
    rng: np.random.Generator = np.random.default_rng()

    def evolve(self, evaluator: Callable[[np.ndarray], float]) -> Tuple[np.ndarray, float]:
        states = [self.rng.normal(scale=0.1, size=self.grid_shape) for _ in range(self.population)]
        scores = [evaluator(state) for state in states]
        for _ in range(self.generations):
            ranked = sorted(zip(scores, states), key=lambda kv: kv[0], reverse=True)
            elites = [state for _, state in ranked[: self.elite]]
            new_states = elites.copy()
            while len(new_states) < self.population:
                parent = self.rng.choice(elites)
                mutant = parent + self.rng.normal(scale=self.mutation_scale, size=self.grid_shape)
                new_states.append(mutant)
            states = new_states
            scores = [evaluator(state) for state in states]
        best_idx = int(np.argmax(scores))
        return states[best_idx], float(scores[best_idx])
