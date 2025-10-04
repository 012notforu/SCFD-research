from __future__ import annotations

import argparse
import math
from typing import Dict

import numpy as np

from engine import accel_theta, extract_symbols, metropolis_accept
from engine.diagnostics import coherence_metrics
from engine.integrators import leapfrog_step
from engine.scheduler import AsyncScheduler
from observe.trackers import SpectrumTracker, SymbolTracker
from run.common import (
    finalize_plots,
    initialize_state,
    load_simulation_config,
    setup_logger,
    store_spectrum,
    summarize_energy,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hybrid SCFD simulation with gate + symbols")
    parser.add_argument("--cfg", default="cfg/defaults.yaml")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    return parser.parse_args()


def _validate_cfl(config) -> None:
    c2 = config.physics.wave_speed_sq
    if c2 <= 0.0:
        raise ValueError(
            f"Effective wave speed squared must be positive; got {c2:.6f}. Adjust gamma/alpha/epsilon."
        )
    dx = config.grid.spacing
    dt = config.integration.dt
    limit = config.integration.cfl_limit * dx / math.sqrt(c2)
    if dt > limit:
        raise ValueError(
            f"Time step {dt:.6f} exceeds CFL limit {limit:.6f} for wave speed sqrt(c^2)={math.sqrt(c2):.6f}"
        )


def main() -> None:
    args = _parse_args()
    config = load_simulation_config(args.cfg)
    steps = args.steps or config.run.steps
    if args.seed is not None:
        config.run.seed = args.seed

    _validate_cfl(config)

    state = initialize_state(config, config.run.seed)
    logger = setup_logger(config, args.outdir)
    print(config.startup_summary())

    dt = config.integration.dt
    dx = config.grid.spacing
    noise_cfg = {
        "enabled": config.integration.noise.enabled,
        "sigma_scale": config.integration.noise.sigma_scale,
        "seed": config.integration.noise.seed,
    }
    energy_series: list[Dict[str, float]] = []
    spectrum_tracker = SpectrumTracker(window=config.observation.trackers.get("spectrum_window", 256))
    symbol_tracker = SymbolTracker(history_window=config.symbolizer.get("history_window", 128))
    temperature = config.free_energy_gate.temperature.min
    scheduler = AsyncScheduler(config.scheduler, config.grid, seed=config.run.seed)

    theta = state["theta"]
    theta_dot = state["theta_dot"]

    def accel(field: np.ndarray) -> np.ndarray:
        return accel_theta(field, config.physics, dx=dx)

    prev_energy = summarize_energy(theta, theta_dot, config, dx=dx)["total"]

    for step in range(steps):
        mask = scheduler.sample_mask(dt)
        theta, theta_dot, _, info = leapfrog_step(
            theta,
            theta_dot,
            accel,
            dt,
            noise_cfg=noise_cfg,
            max_step=config.integration.nudges.max_step,
        )
        metrics = coherence_metrics(theta, config.physics, dx=dx)
        energy = summarize_energy(theta, theta_dot, config, dx=dx)
        energy_series.append(energy)
        delta_energy = energy["total"] - prev_energy
        prev_energy = energy["total"]
        accept_prob = metropolis_accept(
            np.array([delta_energy]),
            temperature,
            config.free_energy_gate.metropolis.clip,
        )[0]
        temperature = float(
            np.clip(
                temperature * (1.0 + 0.05 * (accept_prob - 0.5)),
                config.free_energy_gate.temperature.min,
                config.free_energy_gate.temperature.max,
            )
        )
        symbols = extract_symbols(theta, config)
        spectrum = spectrum_tracker.update(theta)[1]
        symbol_ids = [sym.label for sym in symbols["symbols"]]
        sym_metrics = symbol_tracker.update(symbol_ids)
        logger.log_csv(
            "energy",
            [
                "step",
                "total",
                "kinetic",
                "gradient",
                "coherence",
                "potential",
                "curvature",
                "cross",
                "accept",
            ],
            [
                step,
                energy["total"],
                energy["kinetic"],
                energy["gradient"],
                energy["coherence"],
                energy["potential"],
                energy["curvature"],
                energy["cross"],
                accept_prob,
            ],
        )
        logger.log_step(
            {
                "step": step,
                "energy_total": energy["total"],
                "accept_prob": float(accept_prob),
                "temperature": temperature,
                "spectrum_width": spectrum_tracker.width,
                "symbol_rate": sym_metrics.get("rate", 0.0),
                "symbol_perplexity": sym_metrics.get("perplexity", 0.0),
                "scheduler_density": float(mask.mean()),
                "rms_step": info["rms_step"],
                "fraction_supercritical": metrics["fraction_supercritical"],
            }
        )
    spec = store_spectrum(theta, logger, dx=dx)
    finalize_plots(logger, energy_series, spec)


if __name__ == "__main__":
    main()
