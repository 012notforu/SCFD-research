from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from benchmarks.heat_diffusion import (
    HeatDiffusionControlConfig,
    HeatDiffusionController,
)
from benchmarks.heat_diffusion_mobile import HeatMobileParams, HeatMobileSimulator

DEFAULT_PATH_SPEC = "48,16;48,48;48,80"


def _vector_from_config(control: HeatDiffusionControlConfig, params: HeatMobileParams) -> np.ndarray:
    return np.array(
        [
            control.encode_gain,
            control.encode_decay,
            control.control_gain,
            control.control_clip,
            control.smooth_lambda,
            control.theta_clip,
            params.heater_amplitude,
            float(params.heater_radius),
            params.alpha,
        ],
        dtype=np.float32,
    )


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def _config_from_vector(
    base_control: HeatDiffusionControlConfig,
    base_params: HeatMobileParams,
    vector: np.ndarray,
) -> Tuple[HeatDiffusionControlConfig, HeatMobileParams]:
    vec = np.asarray(vector, dtype=np.float32)
    encode_gain = _clip(vec[0], 0.0, 5.0)
    encode_decay = _clip(vec[1], 0.5, 0.999)
    control_gain = _clip(vec[2], 1e-5, 1e-2)
    control_clip = _clip(vec[3], 1e-4, 0.15)
    smooth_lambda = _clip(vec[4], 0.0, 1.0)
    theta_clip = _clip(vec[5], 0.5, 5.0)
    heater_amp = _clip(vec[6], 0.0, 1.5)
    heater_radius = int(np.clip(np.round(vec[7]), 1, 10))
    alpha = _clip(vec[8], 0.05, 0.35)

    control = replace(
        base_control,
        encode_gain=encode_gain,
        encode_decay=encode_decay,
        control_gain=control_gain,
        control_clip=control_clip,
        smooth_lambda=smooth_lambda,
        theta_clip=theta_clip,
    )
    params = replace(
        base_params,
        heater_amplitude=heater_amp,
        heater_radius=heater_radius,
        alpha=alpha,
    )
    return control, params


def _parse_path(spec: str | None, shape: Tuple[int, int]) -> Tuple[Tuple[int, int], ...]:
    if not spec:
        return HeatMobileParams().path
    coords: List[Tuple[int, int]] = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid waypoint '{chunk}'. Expected 'row,col'.")
        y, x = (int(float(parts[0])), int(float(parts[1])))
        h, w = shape
        coords.append((int(np.clip(y, 0, h - 1)), int(np.clip(x, 0, w - 1))))
    if not coords:
        raise ValueError("Parsed waypoint list is empty.")
    return tuple(coords)


def _evaluate(
    control_cfg: HeatDiffusionControlConfig,
    params: HeatMobileParams,
    seeds: Iterable[int],
    steps: int,
    record_interval: int,
    target_kind: str,
) -> Tuple[float, Dict[str, float]]:
    mses: List[float] = []
    energies: List[float] = []
    budgets: List[float] = []
    progress: List[float] = []
    for seed in seeds:
        sim_params = replace(params, init_seed=int(seed))
        controller = HeatDiffusionController(control_cfg, sim_params.shape)
        simulator = HeatMobileSimulator(sim_params, controller, target_kind=target_kind)
        history = simulator.run(steps=steps, record_interval=record_interval)
        last = history["metrics"][-1]
        mses.append(float(last["mse"]))
        energies.append(float(last["energy"]))
        budgets.append(float(last["budget_util"]))
        progress.append(float(last["path_progress"]))
    mse_arr = np.array(mses, dtype=np.float32)
    energy_arr = np.array(energies, dtype=np.float32)
    budget_arr = np.array(budgets, dtype=np.float32)
    prog_arr = np.array(progress, dtype=np.float32)
    metrics = {
        "mean_mse": float(mse_arr.mean()),
        "std_mse": float(mse_arr.std()),
        "mean_energy": float(energy_arr.mean()),
        "mean_budget": float(budget_arr.mean()),
        "mean_progress": float(prog_arr.mean()),
    }
    score = -metrics["mean_mse"] - 0.05 * metrics["mean_budget"] + 0.02 * metrics["mean_progress"]
    return score, metrics


def _write_history_header(path: Path) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "generation",
                "candidate",
                "score",
                "mean_mse",
                "std_mse",
                "mean_energy",
                "mean_budget",
                "mean_progress",
                "is_best",
            ]
        )


def _append_history(
    path: Path,
    generation: int,
    candidate: int,
    score: float,
    metrics: Dict[str, float],
    is_best: bool,
) -> None:
    with path.open("a", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                generation,
                candidate,
                score,
                metrics["mean_mse"],
                metrics["std_mse"],
                metrics["mean_energy"],
                metrics["mean_budget"],
                metrics["mean_progress"],
                int(is_best),
            ]
        )


def _save_vector(path: Path, vector: np.ndarray, metrics: Dict[str, float]) -> None:
    path.write_text(json.dumps({"vector": vector.tolist(), "metrics": metrics}, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CMA search for mobile heat actuator benchmark")
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--elite", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=0.3)
    parser.add_argument("--sigma-decay", type=float, default=0.95)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--record-interval", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--path", type=str, default=DEFAULT_PATH_SPEC, help="Waypoints as 'y,x;...'")
    parser.add_argument("--target", type=str, default="moving_gaussian", choices=["moving_gaussian", "gradient", "hot_corner"])
    parser.add_argument("--outdir", type=str, default="runs/heat_mobile_cma")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.elite <= 0 or args.elite > args.population:
        raise ValueError("elite must be in [1, population]")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    history_path = outdir / "history.csv"
    _write_history_header(history_path)

    rng = np.random.default_rng(args.seed)
    base_control = HeatDiffusionControlConfig()
    base_params = HeatMobileParams()
    base_params = replace(base_params, path=_parse_path(args.path, base_params.shape))
    mean = _vector_from_config(base_control, base_params)
    sigma = float(args.sigma)

    seeds = rng.integers(0, 2**32 - 1, size=args.episodes, endpoint=False)

    best_score = -np.inf
    best_vector = mean.copy()
    best_metrics: Dict[str, float] = {}

    dim = mean.size

    for gen in range(args.generations):
        candidates: List[Tuple[float, np.ndarray, Dict[str, float]]] = []

        control_cfg, params = _config_from_vector(base_control, base_params, mean)
        score, metrics = _evaluate(control_cfg, params, seeds, args.steps, args.record_interval, args.target)
        candidates.append((score, mean.copy(), metrics))
        _append_history(history_path, gen, 0, score, metrics, score > best_score)

        for cand_idx in range(1, args.population):
            sample = mean + sigma * rng.normal(size=dim)
            control_cfg, params = _config_from_vector(base_control, base_params, sample)
            score, metrics = _evaluate(control_cfg, params, seeds, args.steps, args.record_interval, args.target)
            candidates.append((score, sample, metrics))
            _append_history(history_path, gen, cand_idx, score, metrics, score > best_score)

        candidates.sort(key=lambda item: item[0], reverse=True)
        elites = candidates[: args.elite]

        elite_vectors = np.stack([vec for _, vec, _ in elites], axis=0)
        mean = elite_vectors.mean(axis=0)
        control_cfg, params = _config_from_vector(base_control, base_params, mean)
        mean = _vector_from_config(control_cfg, params)

        if elites[0][0] > best_score:
            best_score, best_vector, best_metrics = elites[0]
            _save_vector(outdir / "best_vector.json", best_vector, best_metrics)

        sigma *= args.sigma_decay
        print(
            f"[gen {gen:03d}] best_score={elites[0][0]:.6f} mean_mse={elites[0][2]['mean_mse']:.6f} "
            f"budget={elites[0][2]['mean_budget']:.3f} progress={elites[0][2]['mean_progress']:.3f} "
            f"global_best={-best_score:.6f} mse, sigma={sigma:.4f}",
            flush=True,
        )

    final_control, final_params = _config_from_vector(base_control, base_params, best_vector)
    controller = HeatDiffusionController(final_control, final_params.shape)
    simulator = HeatMobileSimulator(final_params, controller, target_kind=args.target)
    history = simulator.run(steps=args.steps, record_interval=args.record_interval)
    artifact_dir = outdir / "best_artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.save(artifact_dir / "metrics.npy", np.array(history["metrics"], dtype=object))
    vis = simulator.generate_visualization(artifact_dir, history)
    _save_vector(outdir / "best_vector.json", best_vector, best_metrics)
    print("Training complete. Best results saved in", outdir)
    for key, value in vis.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
