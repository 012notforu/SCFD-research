from __future__ import annotations

import argparse
import math
from typing import Dict

import numpy as np

from engine.integrators import leapfrog_step
from engine.diagnostics import coherence_metrics, energy_drift
from engine import accel_theta
from observe.trackers import SpectrumTracker
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
    parser = argparse.ArgumentParser(description="Run conservative SCFD simulation")
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
    if args.seed is not None:
        config.run.seed = args.seed
    steps = args.steps or config.run.steps

    _validate_cfl(config)

    state = initialize_state(config, config.run.seed)
    logger = setup_logger(config, args.outdir)

    c2 = config.physics.wave_speed_sq
    margin = config.physics.coherence_margin()
    dt = config.integration.dt
    dx = config.grid.spacing
    cfl_ratio = math.sqrt(max(c2, 0.0)) * dt / dx
    print(config.startup_summary())
    print(f"[CFL] wave_speed_sq={c2:.6f}, margin={margin:.3f}, CFL_ratio={cfl_ratio:.6f}")

    if config.integration.noise.enabled:
        raise ValueError("Conservative run must disable stochastic noise")
    if config.integration.nudges.enable_controller:
        raise ValueError("Controller nudges must be disabled for conservative baseline runs")

    theta = state["theta"]
    theta_dot = state["theta_dot"]

    energy_series: list[Dict[str, float]] = []
    spectrum = (np.zeros(1), np.zeros(1))
    energy_totals: list[float] = []
    clamp_events = 0

    def accel(field: np.ndarray) -> np.ndarray:
        return accel_theta(field, config.physics, dx=dx)

    spectrum_tracker = SpectrumTracker(window=config.observation.trackers.get("spectrum_window", 256))
    impulse_probe = ImpulseProbe(accel, dt)
    last_impulse = {"impulse_peak": 0.0, "impulse_decay": 0.0}

    for step in range(steps):
        theta, theta_dot, _, info = leapfrog_step(
            theta,
            theta_dot,
            accel,
            dt,
            noise_cfg={"enabled": False},
            max_step=None,
        )
        metrics = coherence_metrics(theta, config.physics, dx=dx)
        energy = summarize_energy(theta, theta_dot, config, dx=dx)
        edge_diag = compute_edge_diagnostics(theta, dx)
        centers, spectrum_avg = spectrum_tracker.update(theta)
        hi_freq = high_frequency_fraction(centers, spectrum_avg)
        energy_series.append(energy)
        energy_totals.append(energy["total"])
        if info["clamp_triggered"]:
            clamp_events += 1

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
            ],
        )
        logger.log_step(
            {
                "step": step,
                "energy_total": energy["total"],
                "rms_accel": info["rms_accel"],
                "rms_step": info["rms_step"],
                "rms_velocity_half": info["rms_velocity_half"],
                "fraction_supercritical": metrics["fraction_supercritical"],
                "f_mean": metrics["f_mean"],
                "f_max": metrics["f_max"],
                "edge_density": edge_diag["edge_density"],
                "curvature_stress": edge_diag["curvature_stress"],
                "spectrum_width": spectrum_tracker.width,
                "hi_freq_fraction": hi_freq,
                "impulse_peak": last_impulse["impulse_peak"],
                "impulse_decay": last_impulse["impulse_decay"],
            }
        )
    spectrum = store_spectrum(theta, logger, dx=dx)
    finalize_plots(logger, energy_series, spectrum)
    drift = energy_drift(np.asarray(energy_totals))
    print(f"[Energy] relative drift {drift:.3e} over {steps} steps; clamp events={clamp_events}")


if __name__ == "__main__":
    main()
