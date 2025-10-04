from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from benchmarks.flow_cylinder import FlowCylinderControlConfig
from benchmarks.flow_regime_sweep import FlowRegimeParams, FlowRegimeSweep


def _vector_from_config(control: FlowCylinderControlConfig, params: FlowRegimeParams) -> np.ndarray:
    return np.array(
        [
            control.encode_gain,
            control.encode_decay,
            control.control_gain,
            control.control_clip,
            control.smooth_lambda,
            control.theta_clip,
            params.viscosity,
        ],
        dtype=np.float32,
    )


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def _config_from_vector(
    base_control: FlowCylinderControlConfig,
    base_params: FlowRegimeParams,
    vector: np.ndarray,
) -> Tuple[FlowCylinderControlConfig, FlowRegimeParams]:
    vec = np.asarray(vector, dtype=np.float32)
    encode_gain = _clip(vec[0], 0.0, 5.0)
    encode_decay = _clip(vec[1], 0.5, 0.999)
    control_gain = _clip(vec[2], 1e-4, 5e-2)
    control_clip = _clip(vec[3], 1e-3, 0.2)
    smooth_lambda = _clip(vec[4], 0.0, 1.0)
    theta_clip = _clip(vec[5], 0.5, 5.0)
    viscosity = _clip(vec[6], 0.01, 0.05)
    control = replace(
        base_control,
        encode_gain=encode_gain,
        encode_decay=encode_decay,
        control_gain=control_gain,
        control_clip=control_clip,
        smooth_lambda=smooth_lambda,
        theta_clip=theta_clip,
    )
    params = replace(base_params, viscosity=viscosity)
    return control, params


def _evaluate(
    control_cfg: FlowCylinderControlConfig,
    params: FlowRegimeParams,
    seeds: Iterable[int],
) -> Tuple[float, Dict[str, float]]:
    sweep = FlowRegimeSweep(params, control_cfg)
    sweep.run(seeds=seeds)
    metrics = sweep.aggregate_metrics()
    score = metrics["mean_throughput"] if "mean_throughput" in metrics else 0.0
    # For regime sweep we prioritise low wake error and drag
    score = -(metrics["mean_wake_mse"] + 0.05 * metrics["mean_drag_proxy"])
    return score, metrics


def _write_history_header(path: Path) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "generation",
                "candidate",
                "score",
                "mean_wake_mse",
                "std_wake_mse",
                "mean_drag_proxy",
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
                metrics.get("mean_wake_mse", float("nan")),
                metrics.get("std_wake_mse", float("nan")),
                metrics.get("mean_drag_proxy", float("nan")),
                int(is_best),
            ]
        )


def _save_vector(path: Path, vector: np.ndarray, metrics: Dict[str, float]) -> None:
    data = {
        "vector": vector.tolist(),
        "metrics": metrics,
    }
    path.write_text(json.dumps(data, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CMA search for flow regime sweep controller")
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--elite", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--sigma-decay", type=float, default=0.95)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=800, help="Steps per regime")
    parser.add_argument("--record-interval", type=int, default=40)
    parser.add_argument("--inflows", type=float, nargs="+", default=[0.6, 1.0, 1.4])
    parser.add_argument("--outdir", type=str, default="runs/flow_regime_cma")
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
    base_control = FlowCylinderControlConfig()
    base_params = FlowRegimeParams(
        inflow_values=tuple(args.inflows),
        steps_per_regime=args.steps,
        record_interval=args.record_interval,
    )
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
        score, metrics = _evaluate(control_cfg, params, seeds)
        candidates.append((score, mean.copy(), metrics))
        _append_history(history_path, gen, 0, score, metrics, score > best_score)

        for cand_idx in range(1, args.population):
            sample = mean + sigma * rng.normal(size=dim)
            control_cfg, params = _config_from_vector(base_control, base_params, sample)
            score, metrics = _evaluate(control_cfg, params, seeds)
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
            f"[gen {gen:03d}] best_score={elites[0][0]:.6f} mean_wake_mse={elites[0][2].get('mean_wake_mse', float('nan')):.6f} global_best={best_score:.6f} score, sigma={sigma:.4f}",
            flush=True,
        )

    final_control, final_params = _config_from_vector(base_control, base_params, best_vector)
    sweep = FlowRegimeSweep(final_params, final_control)
    sweep.run(seeds=seeds)
    artifact_dir = outdir / "best_artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics = sweep.regime_metrics
    np.save(artifact_dir / "metrics.npy", np.array(metrics, dtype=object))
    vis = sweep.save_visualizations(artifact_dir)
    _save_vector(outdir / "best_vector.json", best_vector, best_metrics)
    print("Training complete. Best results saved in", outdir)
    for key, value in vis.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
