from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from engine import (
    compute_energy_report,
    load_config,
    metropolis_accept,
)
from engine.diagnostics import radial_spectrum
from engine.params import SimulationConfig
from utils.logging import RunLogger, create_run_directory
from utils.viz import plot_energy_breakdown, plot_spectrum


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
