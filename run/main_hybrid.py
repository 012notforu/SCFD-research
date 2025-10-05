from __future__ import annotations

import argparse
import math
from collections import deque
from typing import Dict

import numpy as np

from engine import accel_theta, extract_symbols, metropolis_accept
from engine.diagnostics import coherence_metrics
from engine.integrators import leapfrog_step
from engine.scheduler import AsyncScheduler
from observe.trackers import SpectrumTracker, SymbolTracker, HorizonTracker
from run.common import (
    ImpulseProbe,
    compute_edge_diagnostics,
    finalize_plots,
    high_frequency_fraction,
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
    horizon_tracker = HorizonTracker(threshold=config.observation.trackers.get("horizon_threshold", 1e-2))
    temperature = config.free_energy_gate.temperature.min
    scheduler = AsyncScheduler(config.scheduler, config.grid, seed=config.run.seed)

    theta = state["theta"]
    theta_dot = state["theta_dot"]

    def accel(field: np.ndarray) -> np.ndarray:
        return accel_theta(field, config.physics, dx=dx)

    impulse_probe = ImpulseProbe(accel, dt)
    last_impulse = {"impulse_peak": 0.0, "impulse_decay": 0.0}
    gate_history: deque[float] = deque(maxlen=512)

    rng_twin = np.random.default_rng(config.run.seed + 1)
    theta_twin = theta + 1e-5 * rng_twin.normal(size=theta.shape)
    theta_dot_twin = theta_dot.copy()

    prev_energy = summarize_energy(theta, theta_dot, config, dx=dx)["total"]

    for step in range(steps):
        mask = scheduler.sample_mask(dt)
        synchrony = float(mask.mean())
        theta, theta_dot, _, info = leapfrog_step(
            theta,
            theta_dot,
            accel,
            dt,
            noise_cfg=noise_cfg,
            max_step=config.integration.nudges.max_step,
        )
        theta_twin, theta_dot_twin, _, _ = leapfrog_step(
            theta_twin,
            theta_dot_twin,
            accel,
            dt,
            noise_cfg={"enabled": False},
            max_step=config.integration.nudges.max_step,
        )
        horizon_estimate = horizon_tracker.update(theta, theta_twin)
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
        centers, spectrum_avg = spectrum_tracker.update(theta)
        hi_freq = high_frequency_fraction(centers, spectrum_avg)
        symbol_ids = [sym.label for sym in symbols["symbols"]]
        sym_metrics = symbol_tracker.update(symbol_ids)
        edge_diag = compute_edge_diagnostics(theta, dx)
        gate_history.append(float(accept_prob))
        hist_counts, _ = np.histogram(gate_history, bins=np.linspace(0.0, 1.0, 6))
        hist_counts = hist_counts / max(len(gate_history), 1)
        gate_hist_low = float(hist_counts[0])
        gate_hist_high = float(hist_counts[-1])

        if config.logging.impulse_interval > 0 and (step + 1) % config.logging.impulse_interval == 0:
            last_impulse = impulse_probe.measure(theta, theta_dot)

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
                "scheduler_density": synchrony,
                "rms_step": info["rms_step"],
                "fraction_supercritical": metrics["fraction_supercritical"],
                "edge_density": edge_diag["edge_density"],
                "curvature_stress": edge_diag["curvature_stress"],
                "hi_freq_fraction": hi_freq,
                "impulse_peak": last_impulse["impulse_peak"],
                "impulse_decay": last_impulse["impulse_decay"],
                "horizon_steps": horizon_estimate,
                "gate_hist_low": gate_hist_low,
                "gate_hist_high": gate_hist_high,
            }
        )
    spec = store_spectrum(theta, logger, dx=dx)
    finalize_plots(logger, energy_series, spec)


if __name__ == "__main__":
    main()
