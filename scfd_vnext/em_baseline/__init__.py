"""Emergent Models baseline implementation."""

from .transition_f import LeniaTransition
from .encode_decode import encode_input, decode_output
from .halting import HaltingController
from .optimizer import RandomStateOptimizer
from .runner import run_em_episode
from .diagnostics_em import compute_symbol_metrics

__all__ = [
    "LeniaTransition",
    "encode_input",
    "decode_output",
    "HaltingController",
    "RandomStateOptimizer",
    "run_em_episode",
    "compute_symbol_metrics",
]
