from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

from benchmarks.heat_diffusion import (
    HeatDiffusionControlConfig,
    HeatDiffusionController,
    HeatDiffusionParams,
    HeatDiffusionSimulator,
    synthetic_temperature,
)
from benchmarks.flow_redundant import FlowRedundantParams, FlowRedundantSimulator
from benchmarks.flow_cylinder import FlowCylinderControlConfig, FlowCylinderController
from benchmarks.wave_field import WaveFieldControlConfig, WaveFieldController, synthetic_wave_target
from benchmarks.wave_field_mode_switch import WaveModeSwitchParams, WaveModeSwitchSimulator
from run.train_cma_heat import _config_from_vector as heat_vector_to_cfg
from run.train_cma_flow_redundant import _config_from_vector as flow_vector_to_cfg
from run.train_cma_wave_mode_switch import _config_from_vector as wave_vector_to_cfg


Numeric = (int, float, np.floating, np.integer)


def _profile_steps(simulator, steps: int, warmup: int) -> Tuple[Dict[str, object], Dict[str, float]]:
    elapsed: list[float] = []
    final_stats: Dict[str, float] = {}
    for step in range(steps):
        tic = time.perf_counter()
        stats = simulator.step()
        toc = time.perf_counter()
        filtered = {key: float(value) for key, value in stats.items() if isinstance(value, Numeric)}
        final_stats = filtered
        if step >= warmup:
            elapsed.append((toc - tic) * 1000.0)
    profile: Dict[str, object] = {
        "steps_measured": max(0, len(elapsed)),
        "warmup_steps": warmup,
    }
    if elapsed:
        profile["avg_step_ms"] = float(statistics.mean(elapsed))
        profile["std_step_ms"] = float(statistics.pstdev(elapsed))
        profile["max_step_ms"] = float(max(elapsed))
    else:
        profile["avg_step_ms"] = profile["std_step_ms"] = profile["max_step_ms"] = 0.0
    return profile, final_stats


def profile_heat_diffusion(
    vector_path: str | Path | None = None,
    *,
    steps: int = 400,
    warmup: int = 40,
    seed: int = 0,
) -> Dict[str, object]:
    vector_path = Path(vector_path or "runs/heat_diffusion_cma_hotcorner/best_vector.json")
    data = json.loads(vector_path.read_text())
    vector = np.asarray(data["vector"], dtype=np.float32)

    base_control = HeatDiffusionControlConfig()
    base_params = HeatDiffusionParams()
    control_cfg, params = heat_vector_to_cfg(base_control, base_params, vector)
    params = replace(params, init_seed=seed)
    controller = HeatDiffusionController(control_cfg, params.shape)
    target = synthetic_temperature(params.shape, kind="hot_corner")
    simulator = HeatDiffusionSimulator(params, controller, target)

    profile, final_stats = _profile_steps(simulator, steps, warmup)

    control = np.tanh(simulator.controller.theta)
    delta = np.clip(control_cfg.control_gain * control, -control_cfg.control_clip, control_cfg.control_clip)
    profile["mean_abs_delta"] = float(np.mean(np.abs(delta)))
    profile["vector_path"] = str(vector_path)
    profile["vector_metrics"] = data.get("metrics", {})
    profile["final_stats"] = final_stats
    return profile


def profile_flow_redundant(
    vector_path: str | Path | None = None,
    *,
    steps: int = 1400,
    warmup: int = 100,
    seed: int = 0,
) -> Dict[str, object]:
    vector_path = Path(vector_path or "runs/flow_redundant_cma/best_vector.json")
    data = json.loads(vector_path.read_text())
    vector = np.asarray(data["vector"], dtype=np.float32)

    base_control = FlowCylinderControlConfig()
    base_params = FlowRedundantParams()
    control_cfg, params = flow_vector_to_cfg(base_control, base_params, vector)
    params = replace(params, init_seed=seed)
    controller = FlowCylinderController(control_cfg, params.shape)
    simulator = FlowRedundantSimulator(params, controller)

    profile, final_stats = _profile_steps(simulator, steps, warmup)

    profile["final_throughput"] = float(final_stats.get("throughput", float("nan")))
    profile["final_budget_util"] = float(final_stats.get("budget_util", float("nan")))
    profile["final_actuator_rms"] = float(final_stats.get("actuator_rms", float("nan")))
    profile["vector_path"] = str(vector_path)
    profile["vector_metrics"] = data.get("metrics", {})
    profile["final_stats"] = final_stats
    return profile


def profile_wave_mode_switch(
    vector_path: str | Path | None = None,
    *,
    steps: int = 1600,
    warmup: int = 120,
    seed: int = 0,
) -> Dict[str, object]:
    vector_path = Path(vector_path or "runs/wave_mode_switch_cma/best_vector.json")
    data = json.loads(vector_path.read_text())
    vector = np.asarray(data["vector"], dtype=np.float32)

    base_control = WaveFieldControlConfig()
    base_params = WaveModeSwitchParams()
    control_cfg, params = wave_vector_to_cfg(base_control, base_params, vector)
    params = replace(params, init_seed=seed)
    controller = WaveFieldController(control_cfg, params.shape)
    simulator = WaveModeSwitchSimulator(params, controller)

    profile, final_stats = _profile_steps(simulator, steps, warmup)

    control = np.tanh(simulator.controller.theta)
    delta = np.clip(control_cfg.control_gain * control, -control_cfg.control_clip, control_cfg.control_clip)
    boundary_delta = delta[simulator.boundary_mask]
    profile["boundary_mean_abs_delta"] = float(np.mean(np.abs(boundary_delta)))
    profile["vector_path"] = str(vector_path)
    profile["vector_metrics"] = data.get("metrics", {})
    profile["final_stats"] = final_stats
    profile["final_phase"] = final_stats.get("phase")
    return profile


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile latency and budget usage for SCFD controllers")
    parser.add_argument(
        "--domains",
        nargs="*",
        default=["heat", "flow", "wave"],
        choices=["heat", "flow", "wave"],
        help="Domains to profile (default: heat flow wave)",
    )
    parser.add_argument("--steps", type=int, default=800, help="Steps per profile run")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup steps to ignore in timing")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="", help="Optional JSON output path")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    results: Dict[str, object] = {}
    if "heat" in args.domains:
        results["heat_diffusion"] = profile_heat_diffusion(steps=args.steps, warmup=args.warmup, seed=args.seed)
    if "flow" in args.domains:
        results["flow_redundant"] = profile_flow_redundant(steps=max(args.steps, 600), warmup=args.warmup, seed=args.seed)
    if "wave" in args.domains:
        results["wave_mode_switch"] = profile_wave_mode_switch(steps=max(args.steps, 800), warmup=args.warmup, seed=args.seed)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"Latency profile written to {args.out}")
    else:
        for domain, payload in results.items():
            print(f"=== {domain} ===")
            for key, value in payload.items():
                if key not in {"vector_metrics", "final_stats"}:
                    print(f"  {key}: {value}")


if __name__ == "__main__":
    main()