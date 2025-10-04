from __future__ import annotations

import argparse
from typing import Dict, Optional

import numpy as np

from engine import accel_theta, extract_symbols
from engine.diagnostics import coherence_metrics
from engine.integrators import leapfrog_step
from observe import (
    FieldAdapter,
    SpectrumTracker,
    SymbolTracker,
    GentleController,
    PrototypeBank,
    NGramLanguageModel,
)
from observe.ae_predictor import AutoEncoderPredictor
from run.common import (
    finalize_plots,
    initialize_state,
    load_simulation_config,
    setup_logger,
    store_spectrum,
    summarize_energy,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SCFD with observation + ML loop")
    parser.add_argument("--cfg", default="cfg/defaults.yaml")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    return parser.parse_args()


def _maybe_autoencoder(adapter: FieldAdapter, latent_dim: int) -> Optional[AutoEncoderPredictor]:
    flat_dim = adapter.stack(np.zeros(adapter.config.grid.shape), np.zeros(adapter.config.grid.shape)).reshape(3, -1).shape[1]
    try:
        return AutoEncoderPredictor(input_dim=flat_dim, latent_dim=latent_dim, hidden_dim=32, lr=1e-3)
    except RuntimeError:
        return None


def main() -> None:
    args = _parse_args()
    config = load_simulation_config(args.cfg)
    steps = args.steps or config.run.steps
    if args.seed is not None:
        config.run.seed = args.seed
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
    controller_cfg = config.observation.controller
    controller = GentleController(
        max_step=controller_cfg.get("output_clamp", 0.1),
        ema_tau=controller_cfg.get("ema_tau", 100.0),
        surprise_threshold=controller_cfg.get("surprise_threshold", 2.0),
    )
    adapter = FieldAdapter(config)
    proto_bank = PrototypeBank()
    lm = NGramLanguageModel(order=3)
    autoencoder = _maybe_autoencoder(adapter, latent_dim=config.observation.autoencoder.get("latent_dim", 8))

    theta = state["theta"]
    theta_dot = state["theta_dot"]

    def accel(field: np.ndarray) -> np.ndarray:
        return accel_theta(field, config.physics, dx=dx)

    prev_energy = summarize_energy(theta, theta_dot, config, dx=dx)["total"]
    prev_theta = theta.copy()

    for step in range(steps):
        theta, theta_dot, _, info = leapfrog_step(
            theta,
            theta_dot,
            accel,
            dt,
            noise_cfg=noise_cfg,
            max_step=config.integration.nudges.max_step,
        )
        energy = summarize_energy(theta, theta_dot, config, dx=dx)
        energy_series.append(energy)
        delta_energy = energy["total"] - prev_energy
        prev_energy = energy["total"]
        metrics_coh = coherence_metrics(theta, config.physics, dx=dx)
        symbols = extract_symbols(theta, config)
        symbol_ids = [sym.label for sym in symbols["symbols"]]
        sym_metrics = symbol_tracker.update(symbol_ids)
        lm.update(symbol_ids)
        obs = adapter.prepare_observation(theta, theta_dot, state["heterogeneity"])
        recon_loss = None
        if autoencoder is not None:
            batch = obs["flat"].astype(np.float32)
            recon_loss = autoencoder.step(batch)
        proto_bank.update(step % 32, obs["raw"][0])  # crude prototype tracking
        diff = np.linalg.norm(theta - prev_theta)
        prev_theta = theta.copy()
        controller_metrics = {
            "spectrum_width": spectrum_tracker.update(theta)[1].mean(),
            "perplexity": sym_metrics.get("perplexity", 0.0),
            "horizon": diff,
            "energy_drift": delta_energy,
        }
        adjustments = controller.step(controller_metrics)
        if "dt_scale" in adjustments:
            dt *= adjustments["dt_scale"]
        config.physics.gamma = float(np.clip(config.physics.gamma + adjustments.get("gamma", 0.0), 0.1, 2.0))
        config.physics.alpha = float(np.clip(config.physics.alpha + adjustments.get("alpha", 0.0), 0.1, 2.0))
        temp_adjust = adjustments.get("T", 0.0)
        config.free_energy_gate.temperature.min = max(0.1, config.free_energy_gate.temperature.min + temp_adjust)
        config.free_energy_gate.temperature.max = max(
            config.free_energy_gate.temperature.min + 0.1,
            config.free_energy_gate.temperature.max + temp_adjust,
        )
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
                "recon_loss",
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
                recon_loss if recon_loss is not None else -1.0,
            ],
        )
        log_payload = {
            "step": step,
            "energy_total": energy["total"],
            "spectrum_width": controller_metrics["spectrum_width"],
            "symbol_perplexity": sym_metrics.get("perplexity", 0.0),
            "controller_T": adjustments.get("T", 0.0),
            "controller_alpha": adjustments.get("alpha", 0.0),
            "controller_gamma": adjustments.get("gamma", 0.0),
            "horizon_proxy": controller_metrics["horizon"],
            "rms_step": info["rms_step"],
            "fraction_supercritical": metrics_coh["fraction_supercritical"],
        }
        if recon_loss is not None:
            log_payload["recon_loss"] = recon_loss
        logger.log_step(log_payload)
    spec = store_spectrum(theta, logger, dx=dx)
    finalize_plots(logger, energy_series, spec)


if __name__ == "__main__":
    main()
