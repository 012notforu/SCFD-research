from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Iterable, List

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
from run.train_cma_heat import _config_from_vector as heat_config_from_vector
from run.train_cma_flow_redundant import _config_from_vector as flow_config_from_vector
from run.train_cma_wave_mode_switch import _config_from_vector as wave_config_from_vector


@dataclass
class Scenario:
    name: str
    description: str
    hook: Callable[[object, Dict[str, float], int, np.random.Generator], None]


def _run_scenario(
    sim_factory: Callable[[], object],
    steps: int,
    record_interval: int,
    hook: Callable[[object, Dict[str, float], int, np.random.Generator], None],
    rng_seed: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(rng_seed)
    simulator = sim_factory()
    metrics: List[Dict[str, float]] = []
    final_stats: Dict[str, float] | None = None
    for step in range(steps):
        stats = simulator.step()
        hook(simulator, stats, step, rng)
        final_stats: Dict[str, object] = {}
        for key, value in stats.items():
            if isinstance(value, (float, int, np.floating, np.integer)):
                final_stats[key] = float(value)
            else:
                final_stats[key] = value
        if (step % record_interval) == 0 or step == steps - 1:
            metrics.append({"step": step, **final_stats})
    assert final_stats is not None
    return {"final": final_stats, "metrics": metrics}


def _summarize_heat(final: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, float]:
    baseline_mse = max(baseline.get("mse", 0.0), 1e-9)
    baseline_energy = max(baseline.get("energy", 0.0), 1e-9)
    return {
        "final_mse": final.get("mse", float("nan")),
        "mse_ratio": final.get("mse", float("nan")) / baseline_mse,
        "final_energy": final.get("energy", float("nan")),
        "energy_ratio": final.get("energy", float("nan")) / baseline_energy,
    }


def _summarize_flow(final: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, float]:
    baseline_throughput = max(baseline.get("throughput", 0.0), 1e-9)
    baseline_energy = max(baseline.get("energy", 0.0), 1e-9)
    return {
        "final_throughput": final.get("throughput", float("nan")),
        "throughput_ratio": final.get("throughput", float("nan")) / baseline_throughput,
        "final_energy": final.get("energy", float("nan")),
        "energy_ratio": final.get("energy", float("nan")) / baseline_energy,
        "final_budget_util": final.get("budget_util", float("nan")),
        "final_actuator_rms": final.get("actuator_rms", float("nan")),
    }


def _summarize_wave(final: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, object]:
    baseline_mse = max(baseline.get("mse", 0.0), 1e-9)
    baseline_energy = max(baseline.get("energy", 0.0), 1e-9)
    summary: Dict[str, object] = {
        "final_mse": final.get("mse", float("nan")),
        "mse_ratio": final.get("mse", float("nan")) / baseline_mse,
        "final_energy": final.get("energy", float("nan")),
        "energy_ratio": final.get("energy", float("nan")) / baseline_energy,
    }
    if "phase" in final:
        summary["phase"] = final["phase"]
    return summary


def evaluate_heat_diffusion(
    vector_path: str | Path | None = None,
    *,
    steps: int = 400,
    record_interval: int = 40,
    seed: int = 0,
    noise_scale: float = 0.02,
    alpha_shift: float = 1.15,
) -> Dict[str, object]:
    vector_path = Path(vector_path or "runs/heat_diffusion_cma_hotcorner/best_vector.json")
    vector_data = json.loads(vector_path.read_text())
    vector = np.asarray(vector_data["vector"], dtype=np.float32)
    base_rng = np.random.default_rng(seed)
    shift_step = max(1, min(steps - 1, steps // 2))

    def sim_factory() -> HeatDiffusionSimulator:
        base_control = HeatDiffusionControlConfig()
        base_params = HeatDiffusionParams()
        control_cfg, params = heat_config_from_vector(base_control, base_params, vector)
        params = replace(params, init_seed=seed)
        controller = HeatDiffusionController(control_cfg, params.shape)
        target = synthetic_temperature(params.shape, kind="hot_corner")
        return HeatDiffusionSimulator(params, controller, target)

    def baseline_hook(sim, stats, step, rng) -> None:  # type: ignore[override]
        return None

    def noise_hook(sim, stats, step, rng) -> None:  # type: ignore[override]
        sim.temp += noise_scale * rng.standard_normal(sim.temp.shape, dtype=np.float32)

    def alpha_hook(sim, stats, step, rng) -> None:  # type: ignore[override]
        if step == shift_step:
            sim.params.alpha *= alpha_shift

    scenarios = [
        ("baseline", "No perturbation", baseline_hook),
        ("noise", f"Add Gaussian field noise (scale={noise_scale})", noise_hook),
        ("alpha_shift", f"Multiply alpha by {alpha_shift:.2f} at step {shift_step}", alpha_hook),
    ]

    results: Dict[str, object] = {
        "vector_path": str(vector_path),
        "vector_metrics": vector_data.get("metrics", {}),
        "scenarios": {},
    }

    baseline_run = _run_scenario(sim_factory, steps, record_interval, baseline_hook, int(base_rng.integers(0, 2**32)))
    baseline_summary = _summarize_heat(baseline_run["final"], baseline_run["final"])
    results["scenarios"]["baseline"] = {
        "description": "No perturbation",
        "summary": baseline_summary,
        "metrics": baseline_run["metrics"],
    }

    for name, description, hook in scenarios[1:]:
        run_result = _run_scenario(sim_factory, steps, record_interval, hook, int(base_rng.integers(0, 2**32)))
        summary = _summarize_heat(run_result["final"], baseline_run["final"])
        results["scenarios"][name] = {
            "description": description,
            "summary": summary,
            "metrics": run_result["metrics"],
        }

    return results


def evaluate_flow_redundant(
    vector_path: str | Path | None = None,
    *,
    steps: int = 1400,
    record_interval: int = 50,
    seed: int = 0,
    noise_scale: float = 0.05,
    inflow_drop: float = 0.85,
) -> Dict[str, object]:
    vector_path = Path(vector_path or "runs/flow_redundant_cma/best_vector.json")
    vector_data = json.loads(vector_path.read_text())
    vector = np.asarray(vector_data["vector"], dtype=np.float32)
    base_rng = np.random.default_rng(seed)
    shift_step = max(1, min(steps - 1, steps // 2))

    def sim_factory() -> FlowRedundantSimulator:
        base_control = FlowCylinderControlConfig()
        base_params = FlowRedundantParams()
        control_cfg, params = flow_config_from_vector(base_control, base_params, vector)
        params = replace(params, init_seed=seed)
        controller = FlowCylinderController(control_cfg, params.shape)
        return FlowRedundantSimulator(params, controller)

    def baseline_hook(sim, stats, step, rng) -> None:  # type: ignore[override]
        return None

    def noise_hook(sim, stats, step, rng) -> None:  # type: ignore[override]
        sim.u += noise_scale * rng.standard_normal(sim.u.shape, dtype=np.float32)
        sim.v += noise_scale * rng.standard_normal(sim.v.shape, dtype=np.float32)

    def inflow_hook(sim, stats, step, rng) -> None:  # type: ignore[override]
        if step == shift_step:
            sim.params.inflow *= inflow_drop

    scenarios = [
        ("baseline", "No perturbation", baseline_hook),
        ("actuator_noise", f"Add Gaussian actuator noise (scale={noise_scale})", noise_hook),
        ("inflow_drop", f"Multiply inflow by {inflow_drop:.2f} at step {shift_step}", inflow_hook),
    ]

    results: Dict[str, object] = {
        "vector_path": str(vector_path),
        "vector_metrics": vector_data.get("metrics", {}),
        "scenarios": {},
    }

    baseline_run = _run_scenario(sim_factory, steps, record_interval, baseline_hook, int(base_rng.integers(0, 2**32)))
    results["scenarios"]["baseline"] = {
        "description": "No perturbation",
        "summary": _summarize_flow(baseline_run["final"], baseline_run["final"]),
        "metrics": baseline_run["metrics"],
    }

    for name, description, hook in scenarios[1:]:
        run_result = _run_scenario(sim_factory, steps, record_interval, hook, int(base_rng.integers(0, 2**32)))
        summary = _summarize_flow(run_result["final"], baseline_run["final"])
        results["scenarios"][name] = {
            "description": description,
            "summary": summary,
            "metrics": run_result["metrics"],
        }

    return results


def evaluate_wave_mode_switch(
    vector_path: str | Path | None = None,
    *,
    steps: int = 1600,
    record_interval: int = 50,
    seed: int = 0,
    noise_scale: float = 0.02,
    switch_delay: int | None = None,
) -> Dict[str, object]:
    vector_path = Path(vector_path or "runs/wave_mode_switch_cma/best_vector.json")
    vector_data = json.loads(vector_path.read_text())
    vector = np.asarray(vector_data["vector"], dtype=np.float32)
    base_rng = np.random.default_rng(seed)
    delay = switch_delay if switch_delay is not None else steps // 6

    def sim_factory() -> WaveModeSwitchSimulator:
        base_control = WaveFieldControlConfig()
        base_params = WaveModeSwitchParams()
        control_cfg, params = wave_config_from_vector(base_control, base_params, vector)
        params = replace(params, init_seed=seed)
        controller = WaveFieldController(control_cfg, params.shape)
        return WaveModeSwitchSimulator(params, controller)

    def baseline_hook(sim, stats, step, rng) -> None:  # type: ignore[override]
        return None

    def noise_hook(sim, stats, step, rng) -> None:  # type: ignore[override]
        if hasattr(sim, "field"):
            sim.field += noise_scale * rng.standard_normal(sim.field.shape, dtype=np.float32)

    def delay_hook(sim, stats, step, rng) -> None:  # type: ignore[override]
        if step == 0:
            sim.params.switch_step += delay
        if step == sim.params.switch_step:
            sim.current_target = synthetic_wave_target(sim.shape, kind=sim.params.switch_kind)

    scenarios = [
        ("baseline", "No perturbation", baseline_hook),
        ("field_noise", f"Add Gaussian field noise (scale={noise_scale})", noise_hook),
        ("switch_delay", f"Delay switch by {delay} steps", delay_hook),
    ]

    results: Dict[str, object] = {
        "vector_path": str(vector_path),
        "vector_metrics": vector_data.get("metrics", {}),
        "scenarios": {},
    }

    baseline_run = _run_scenario(sim_factory, steps, record_interval, baseline_hook, int(base_rng.integers(0, 2**32)))
    results["scenarios"]["baseline"] = {
        "description": "No perturbation",
        "summary": _summarize_wave(baseline_run["final"], baseline_run["final"]),
        "metrics": baseline_run["metrics"],
    }

    for name, description, hook in scenarios[1:]:
        run_result = _run_scenario(sim_factory, steps, record_interval, hook, int(base_rng.integers(0, 2**32)))
        summary = _summarize_wave(run_result["final"], baseline_run["final"])
        results["scenarios"][name] = {
            "description": description,
            "summary": summary,
            "metrics": run_result["metrics"],
        }

    return results


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run robustness battery scenarios for SCFD benchmarks")
    parser.add_argument(
        "--domains",
        nargs="*",
        default=["heat", "flow", "wave"],
        choices=["heat", "flow", "wave"],
        help="Benchmark domains to evaluate (default: heat flow wave)",
    )
    parser.add_argument("--steps", type=int, default=1600, help="Steps per scenario run")
    parser.add_argument("--record-interval", type=int, default=80, help="Metrics recording interval")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for noise scenarios")
    parser.add_argument("--out", type=str, default="", help="Optional JSON output path")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    results: Dict[str, object] = {}

    if "heat" in args.domains:
        results["heat_diffusion"] = evaluate_heat_diffusion(
            steps=max(200, args.steps),
            record_interval=max(20, args.record_interval // 2),
            seed=args.seed,
        )
    if "flow" in args.domains:
        results["flow_redundant"] = evaluate_flow_redundant(
            steps=max(600, args.steps),
            record_interval=max(30, args.record_interval // 2),
            seed=args.seed,
        )
    if "wave" in args.domains:
        results["wave_mode_switch"] = evaluate_wave_mode_switch(
            steps=max(800, args.steps),
            record_interval=args.record_interval,
            seed=args.seed,
        )

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"Robustness results written to {args.out}")
    else:
        for domain, payload in results.items():
            print(f"=== {domain} ===")
            for name, data in payload["scenarios"].items():
                summary = data["summary"]
                print(f"  [{name}] {data['description']}")
                for key, value in summary.items():
                    print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
