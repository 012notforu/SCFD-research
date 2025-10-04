from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import yaml

from engine import accel_theta, extract_symbols, leapfrog_step
from engine.diagnostics import energy_drift, radial_spectrum
from observe.trackers import SpectrumTracker, SymbolTracker
from run.common import (
    initialize_state,
    load_simulation_config,
    summarize_energy,
)
from utils.logging import create_run_directory
from utils.viz import plot_spectrum

from em_baseline.runner import EMRunConfig, run_em_episode
from em_baseline.diagnostics_em import compute_symbol_metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SCFD and EM baseline")
    parser.add_argument("--scfd-cfg", default="cfg/defaults.yaml")
    parser.add_argument("--em-cfg", default="em_baseline/cfg_em.yaml")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--outdir", type=str, default="runs/compare")
    return parser.parse_args()


def _run_scfd(config_path: str, steps: int) -> Dict[str, float]:
    config = load_simulation_config(config_path)
    state = initialize_state(config, config.run.seed)
    spectrum_tracker = SpectrumTracker(window=128)
    symbol_tracker = SymbolTracker(history_window=config.symbolizer.get("history_window", 128))
    theta = state["theta"]
    theta_dot = state["theta_dot"]
    dx = config.grid.spacing
    dt = config.integration.dt

    def accel(field: np.ndarray) -> np.ndarray:
        return accel_theta(field, config.physics, dx=dx)

    energies = []
    sym_metrics: Dict[str, float] = {"perplexity": 0.0, "rate": 0.0}
    for _ in range(steps):
        theta, theta_dot, _, _ = leapfrog_step(
            theta,
            theta_dot,
            accel,
            dt,
            noise_cfg={
                "enabled": config.integration.noise.enabled,
                "sigma_scale": config.integration.noise.sigma_scale,
                "seed": config.integration.noise.seed,
            },
            max_step=config.integration.nudges.max_step,
        )
        energy = summarize_energy(theta, theta_dot, config)
        energies.append(energy["total"])
        spectrum_tracker.update(theta)
        symbol_ids = [sym.label for sym in extract_symbols(theta, config)["symbols"]]
        sym_metrics = symbol_tracker.update(symbol_ids)
    drift = energy_drift(np.array(energies))
    freqs, spectrum = radial_spectrum(theta, dx=dx)
    return {
        "energy_drift": drift,
        "spectrum_width": spectrum_tracker.width,
        "symbol_perplexity": sym_metrics.get("perplexity", 0.0),
        "symbol_rate": sym_metrics.get("rate", 0.0),
        "freqs": freqs,
        "spectrum": spectrum,
    }


def _run_em(config_path: str, steps: int) -> Dict[str, float]:
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
    result = run_em_episode(
        em_config,
        inputs=[0.0] * em_config.grid_shape[0],
        reward=0.0,
        stochastic_prob=cfg.get("stochastic_prob", 0.0),
    )
    symbols = result["symbols"]
    metrics = compute_symbol_metrics(symbols)
    freqs, spectrum = radial_spectrum(result["state"])
    metrics.update({"freqs": freqs, "spectrum": spectrum})
    return metrics


def main() -> None:
    args = _parse_args()
    outdir = create_run_directory("compare", root=args.outdir)
    scfd = _run_scfd(args.scfd_cfg, args.steps)
    em = _run_em(args.em_cfg, args.steps)
    summary_path = outdir / "scfd_vs_em_summary.csv"
    with summary_path.open("w", encoding="utf-8") as fh:
        fh.write("model,energy_drift,spectrum_width,symbol_perplexity,symbol_rate\n")
        fh.write(
            f"SCFD,{scfd['energy_drift']:.6f},{scfd['spectrum_width']:.6f},{scfd['symbol_perplexity']:.6f},{scfd['symbol_rate']:.6f}\n"
        )
        fh.write(
            f"EM,0.0,{0.0:.6f},{em['perplexity']:.6f},{em['rate']:.6f}\n"
        )
    plot_spectrum(scfd["freqs"], scfd["spectrum"], outdir / "plots" / "scfd_spectrum.png")
    plot_spectrum(em["freqs"], em["spectrum"], outdir / "plots" / "em_spectrum.png")
    (outdir / "scfd_metrics.json").write_text(json.dumps(scfd, default=float), encoding="utf-8")
    (outdir / "em_metrics.json").write_text(json.dumps(em, default=float), encoding="utf-8")
    print(f"Comparison report saved to {summary_path}")


if __name__ == "__main__":
    main()


