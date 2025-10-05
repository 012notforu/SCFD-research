from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

from engine import (
    compute_energy_report,
    edge_metrics,
    impulse_response,
    load_config,
    metropolis_accept,
)
from engine.diagnostics import radial_spectrum
from engine.params import SimulationConfig
from utils.logging import RunLogger, create_run_directory
from utils.viz import plot_energy_breakdown, plot_spectrum


@dataclass
class ImpulseProbe:
    accel_fn: Callable[[np.ndarray], np.ndarray]
    dt: float
    amplitude: float = 1e-3
    steps: int = 32

    def measure(self, theta: np.ndarray, theta_dot: np.ndarray) -> Dict[str, float]:
        return impulse_response(
            theta,
            theta_dot,
            self.accel_fn,
            self.dt,
            amplitude=self.amplitude,
            steps=self.steps,
        )


def load_simulation_config(path: str) -> SimulationConfig:
    return load_config(path)


def initialize_state(config: SimulationConfig, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    theta = rng.normal(scale=0.005, size=config.grid.shape)
    theta_dot = np.zeros_like(theta)
    heterogeneity = config.physics.heterogeneity.generate_field(config.grid)
    return {"theta": theta, "theta_dot": theta_dot, "heterogeneity": heterogeneity}


def setup_logger(config: SimulationConfig, outdir: Optional[str] = None) -> RunLogger:
    directory = create_run_directory(config.run.tag, outdir)
    logger = RunLogger(directory)
    cfg_path = Path("cfg/defaults.yaml")
    if cfg_path.exists():
        logger.dump_config(cfg_path.read_text(encoding="utf-8"))
    return logger


def summarize_energy(theta: np.ndarray, theta_dot: np.ndarray, config: SimulationConfig, dx: float | tuple[float, float] | None = None) -> Dict[str, float]:
    spacing = dx if dx is not None else config.grid.spacing
    return compute_energy_report(theta, theta_dot, config.physics, dx=spacing)


def store_spectrum(theta: np.ndarray, logger: RunLogger, dx: float | tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    freqs, spectrum = radial_spectrum(theta, dx=dx if dx is not None else 1.0)
    for f, p in zip(freqs, spectrum):
        logger.log_csv("spectrum", ["freq", "power"], [f, p])
    return freqs, spectrum


def apply_gentle_gate(
    delta_energy: float,
    temperature: float,
    config: SimulationConfig,
) -> float:
    clip = config.free_energy_gate.metropolis.clip
    probs = metropolis_accept(np.array([delta_energy]), temperature, clip)
    return float(probs[0])


def finalize_plots(logger: RunLogger, energy_series: list[Dict[str, float]], spectrum: tuple[np.ndarray, np.ndarray]) -> None:
    keys = energy_series[0].keys()
    energy_avg = {key: float(np.mean([entry[key] for entry in energy_series])) for key in keys}
    plot_energy_breakdown(energy_avg, logger.directory / "plots" / "energy.png")
    freqs, spec = spectrum
    plot_spectrum(freqs, spec, logger.directory / "plots" / "spectrum.png")


def compute_edge_diagnostics(theta: np.ndarray, dx: float, threshold: float = 0.05) -> Dict[str, float]:
    return edge_metrics(theta, dx=dx, grad_threshold=threshold)


def high_frequency_fraction(freqs: np.ndarray, spectrum: np.ndarray, cutoff_ratio: float = 0.5) -> float:
    if freqs.size == 0 or np.max(freqs) <= 0.0:
        return 0.0
    freqs = np.asarray(freqs)
    spectrum = np.asarray(spectrum)
    positive = spectrum > 0.0
    if not np.any(positive):
        return 0.0
    total_power = float(np.sum(spectrum))
    if total_power <= 0.0:
        return 0.0
    cutoff = cutoff_ratio * np.max(freqs)
    mask = freqs >= cutoff
    return float(np.sum(spectrum[mask]) / total_power)
