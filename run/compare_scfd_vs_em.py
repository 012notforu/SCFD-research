from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import yaml

from engine import accel_theta, extract_symbols, leapfrog_step
from engine.diagnostics import energy_drift
from observe.trackers import SpectrumTracker, SymbolTracker, HorizonTracker
from run.common import (
    ImpulseProbe,
    compute_edge_diagnostics,
    high_frequency_fraction,
    initialize_state,
    load_simulation_config,
    summarize_energy,
)
from utils.logging import create_run_directory
from utils.viz import plot_spectrum


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect SCFD metrics and optional EM baseline for reporting")
    parser.add_argument("--scfd-cfg", default="cfg/defaults.yaml")
    parser.add_argument("--em-cfg", default="em_baseline/cfg_em.yaml")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--outdir", type=str, default="runs/compare")
    parser.add_argument("--em-outdir", type=str, default="runs_em/compare")
    parser.add_argument("--include-em", action="store_true", help="Also run the EM baseline")
    return parser.parse_args()


def _run_scfd(config_path: str, steps: int) -> Dict[str, float]:
    config = load_simulation_config(config_path)
    state = initialize_state(config, config.run.seed)
    spectrum_tracker = SpectrumTracker(window=128)
    symbol_tracker = SymbolTracker(history_window=config.symbolizer.get("history_window", 128))
    horizon_tracker = HorizonTracker(threshold=config.observation.trackers.get("horizon_threshold", 1e-2))
    theta = state["theta"]
    theta_dot = state["theta_dot"]
    dx = config.grid.spacing
    dt = config.integration.dt

    def accel(field: np.ndarray) -> np.ndarray:
        return accel_theta(field, config.physics, dx=dx)

    impulse_probe = ImpulseProbe(accel, dt)
    energies = []
    start = time.perf_counter()

    rng_twin = np.random.default_rng(config.run.seed + 101)
    theta_twin = theta + 1e-5 * rng_twin.normal(size=theta.shape)
    theta_dot_twin = theta_dot.copy()

    noise_cfg = {
        "enabled": config.integration.noise.enabled,
        "sigma_scale": config.integration.noise.sigma_scale,
        "seed": config.integration.noise.seed,
    }

    last_hi_freq = 0.0
    horizon_estimate = 0
    edge_diag = {"edge_density": 0.0, "curvature_stress": 0.0}

    for _ in range(steps):
        theta, theta_dot, _, _ = leapfrog_step(
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
        energy = summarize_energy(theta, theta_dot, config)
        energies.append(energy["total"])
        centers, spectrum_avg = spectrum_tracker.update(theta)
        last_hi_freq = high_frequency_fraction(centers, spectrum_avg)
        symbol_ids = [sym.label for sym in extract_symbols(theta, config)["symbols"]]
        symbol_tracker.update(symbol_ids)
        edge_diag = compute_edge_diagnostics(theta, dx)

    runtime = time.perf_counter() - start
    drift = energy_drift(np.array(energies))
    impulse = impulse_probe.measure(theta, theta_dot)
    freqs, spectrum = spectrum_tracker.update(theta)
    sym_metrics = symbol_tracker.update([])
    return {
        "energy_drift": drift,
        "spectrum_width": spectrum_tracker.width,
        "symbol_perplexity": sym_metrics.get("perplexity", 0.0),
        "symbol_rate": sym_metrics.get("rate", 0.0),
        "hi_freq_fraction": last_hi_freq,
        "edge_density": edge_diag.get("edge_density", 0.0),
        "curvature_stress": edge_diag.get("curvature_stress", 0.0),
        "impulse_peak": impulse["impulse_peak"],
        "impulse_decay": impulse["impulse_decay"],
        "horizon_steps": float(horizon_estimate),
        "runtime": runtime,
        "freqs": freqs,
        "spectrum": spectrum,
    }


def _run_em(config_path: str, steps: int) -> Dict[str, float]:
    from em_baseline.runner import EMRunConfig, run_em_episode
    from em_baseline.diagnostics_em import compute_symbol_metrics

    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    cfg = data["em"]
    em_config = EMRunConfig(
        grid_shape=tuple(cfg["grid_shape"]),
        steps=min(steps, cfg["steps"]),
        kernel_radius=cfg["kernel_radius"],
        kernel_sigma=cfg["kernel_sigma"],
        activation=cfg.get("activation", "tanh"),
        halting=cfg.get("halting", {}),
    )
    start = time.perf_counter()
    result = run_em_episode(
        em_config,
        inputs=[0.0] * em_config.grid_shape[0],
        reward=0.0,
        stochastic_prob=cfg.get("stochastic_prob", 0.0),
    )
    runtime = time.perf_counter() - start
    symbols = result["symbols"]
    metrics = compute_symbol_metrics(symbols)
    state = np.asarray(result["state"]).reshape(em_config.grid_shape)
    freqs, spectrum = _compute_em_spectrum(state)
    hi_freq = high_frequency_fraction(freqs, spectrum)
    metrics.update({
        "freqs": freqs,
        "spectrum": spectrum,
        "hi_freq_fraction": hi_freq,
        "runtime": runtime,
    })
    return metrics


def _compute_em_spectrum(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from engine.diagnostics import radial_spectrum

    freqs, spectrum = radial_spectrum(state)
    return freqs, spectrum


def _write_summary(path: Path, rows: list[Dict[str, float]]) -> None:
    headers = [
        "model",
        "energy_drift",
        "spectrum_width",
        "symbol_perplexity",
        "symbol_rate",
        "hi_freq_fraction",
        "edge_density",
        "curvature_stress",
        "horizon_steps",
        "impulse_decay",
        "runtime",
    ]
    with path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(headers) + "\n")
        for row in rows:
            fh.write(",".join(str(row.get(col, 0.0)) for col in headers) + "\n")


def _filter_scalars(data: Dict[str, float]) -> Dict[str, float]:
    return {k: v for k, v in data.items() if not isinstance(v, np.ndarray)}


def main() -> None:
    args = _parse_args()
    scfd_dir = create_run_directory("scfd", root=args.outdir)
    scfd = _run_scfd(args.scfd_cfg, args.steps)
    plot_spectrum(scfd["freqs"], scfd["spectrum"], scfd_dir / "plots" / "scfd_spectrum.png")
    (scfd_dir / "scfd_metrics.json").write_text(json.dumps(_filter_scalars(scfd), default=float), encoding="utf-8")

    rows = [
        {
            "model": "SCFD",
            **_filter_scalars(scfd),
        }
    ]

    if args.include_em:
        em_dir = create_run_directory("em", root=args.em_outdir)
        em = _run_em(args.em_cfg, args.steps)
        plot_spectrum(em["freqs"], em["spectrum"], em_dir / "plots" / "em_spectrum.png")
        (em_dir / "em_metrics.json").write_text(json.dumps(_filter_scalars(em), default=float), encoding="utf-8")
        rows.append(
            {
                "model": "EM",
                "energy_drift": 0.0,
                "spectrum_width": 0.0,
                "symbol_perplexity": em.get("perplexity", 0.0),
                "symbol_rate": em.get("rate", 0.0),
                "hi_freq_fraction": em.get("hi_freq_fraction", 0.0),
                "edge_density": 0.0,
                "curvature_stress": 0.0,
                "horizon_steps": 0.0,
                "impulse_decay": 0.0,
                "runtime": em.get("runtime", 0.0),
            }
        )

    summary_path = scfd_dir / "scfd_vs_em_summary.csv"
    _write_summary(summary_path, rows)
    print(f"SCFD metrics saved to {summary_path}")
    if args.include_em:
        print("EM metrics stored separately; comparison rows available in summary file.")


if __name__ == "__main__":
    main()
