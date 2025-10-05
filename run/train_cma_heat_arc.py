from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np

from benchmarks.heat_diffusion import HeatDiffusionControlConfig, HeatDiffusionController
from benchmarks.heat_diffusion_arc import HeatDiffusionArcParams, HeatDiffusionArcSimulator, build_transform_cycle


TRANSFORMS = (
    "identity",
    "rotate90",
    "rotate180",
    "rotate270",
    "flip_horizontal",
    "flip_vertical",
    "diag",
    "anti_diag",
)


def _vector_from_config(
    control: HeatDiffusionControlConfig,
    params: HeatDiffusionArcParams,
) -> np.ndarray:
    return np.array(
        [
            control.encode_gain,
            control.encode_decay,
            control.control_gain,
            control.control_clip,
            control.smooth_lambda,
            control.theta_clip,
            params.alpha,
            float(params.transform_cycle_interval),
        ],
        dtype=np.float32,
    )


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def _config_from_vector(
    base_control: HeatDiffusionControlConfig,
    base_params: HeatDiffusionArcParams,
    vector: np.ndarray,
) -> Tuple[HeatDiffusionControlConfig, HeatDiffusionArcParams]:
    vec = np.asarray(vector, dtype=np.float32)
    encode_gain = _clip(vec[0], 0.0, 5.0)
    encode_decay = _clip(vec[1], 0.5, 0.999)
    control_gain = _clip(vec[2], 1e-5, 5e-3)
    control_clip = _clip(vec[3], 1e-4, 0.1)
    smooth_lambda = _clip(vec[4], 0.0, 1.0)
    theta_clip = _clip(vec[5], 0.5, 5.0)
    alpha = _clip(vec[6], 0.05, 0.35)
    cycle_interval = int(np.clip(np.round(vec[7]), 1, 200))

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
        alpha=alpha,
        transform_cycle_interval=cycle_interval,
    )
    return control, params


def _metadata_from_params(
    params: HeatDiffusionArcParams,
) -> Dict[str, object]:
    return {
        "transform_cycle": list(params.transform_cycle),
        "transform_cycle_interval": int(params.transform_cycle_interval),
        "base_target_kind": params.base_target_kind,
        "alpha": float(params.alpha),
        "dt": float(params.dt),
        "noise": float(params.noise),
        "shape": list(params.shape),
    }


def _evaluate(
    control_cfg: HeatDiffusionControlConfig,
    params: HeatDiffusionArcParams,
    seeds: Iterable[int],
    steps: int,
    record_interval: int,
) -> Tuple[float, Dict[str, float]]:
    mses: List[float] = []
    errors: List[float] = []
    for seed in seeds:
        sim_params = replace(params, init_seed=int(seed))
        controller = HeatDiffusionController(control_cfg, sim_params.shape)
        simulator = HeatDiffusionArcSimulator(sim_params, controller)
        history = simulator.run(steps=steps, record_interval=record_interval)
        last = history["metrics"][-1]
        mses.append(float(last["mse"]))
        errors.append(float(last["cycle_error"]))
    mse_arr = np.array(mses, dtype=np.float32)
    err_arr = np.array(errors, dtype=np.float32)
    metrics = {
        "mean_mse": float(mse_arr.mean()),
        "std_mse": float(mse_arr.std()),
        "mean_cycle_error": float(err_arr.mean()),
    }
    score = -metrics["mean_mse"] - 0.5 * metrics["mean_cycle_error"]
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
                "mean_cycle_error",
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
                metrics["mean_cycle_error"],
                int(is_best),
            ]
        )


def _save_vector(
    path: Path,
    vector: np.ndarray,
    metrics: Dict[str, float],
    metadata: Optional[Dict[str, object]] = None,
) -> None:
    payload = {
        "vector": vector.tolist(),
        "metrics": metrics,
    }
    if metadata:
        payload.update(metadata)
    path.write_text(json.dumps(payload, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CMA search for ARC-style heat diffusion controller")
    parser.add_argument("--generations", type=int, default=32)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--elite", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--sigma-decay", type=float, default=0.96)
    parser.add_argument("--episodes", type=int, default=3, help="Seeds per evaluation")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--record-interval", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--transform-cycle", type=str, default="identity,rotate90,flip_horizontal,diag")
    parser.add_argument("--cycle-interval", type=int, default=200)
    parser.add_argument("--base-kind", type=str, default="gradient")
    parser.add_argument("--outdir", type=str, default="runs/heat_arc_cma")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.elite <= 0 or args.elite > args.population:
        raise ValueError("elite must be in [1, population]")

    transforms = build_transform_cycle(args.transform_cycle.split(","))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    history_path = outdir / "history.csv"
    _write_history_header(history_path)

    rng = np.random.default_rng(args.seed)
    base_control = HeatDiffusionControlConfig()
    base_params = HeatDiffusionArcParams(
        transform_cycle=transforms,
        transform_cycle_interval=max(1, args.cycle_interval),
        base_target_kind=args.base_kind,
    )
    mean = _vector_from_config(base_control, base_params)
    sigma = float(args.sigma)

    seeds = rng.integers(0, 2**32 - 1, size=args.episodes, endpoint=False)

    best_score = -np.inf
    best_vector = mean.copy()
    best_metrics: Dict[str, float] = {}
    best_control_cfg = base_control
    best_params_cfg = base_params

    dim = mean.size
    for gen in range(args.generations):
        candidates: List[Tuple[float, np.ndarray, Dict[str, float], HeatDiffusionControlConfig, HeatDiffusionArcParams]] = []

        control_cfg, params = _config_from_vector(base_control, base_params, mean)
        score, metrics = _evaluate(
            control_cfg,
            params,
            seeds=seeds,
            steps=args.steps,
            record_interval=args.record_interval,
        )
        candidates.append((score, mean.copy(), metrics, control_cfg, params))
        _append_history(history_path, gen, 0, score, metrics, score > best_score)

        for cand_idx in range(1, args.population):
            sample = mean + sigma * rng.normal(size=dim)
            control_cfg, params = _config_from_vector(base_control, base_params, sample)
            score, metrics = _evaluate(
                control_cfg,
                params,
                seeds=seeds,
                steps=args.steps,
                record_interval=args.record_interval,
            )
            candidates.append((score, sample, metrics, control_cfg, params))
            _append_history(history_path, gen, cand_idx, score, metrics, score > best_score)

        candidates.sort(key=lambda item: item[0], reverse=True)
        elites = candidates[: args.elite]

        elite_vectors = np.stack([vec for _, vec, _, _, _ in elites], axis=0)
        mean = elite_vectors.mean(axis=0)
        control_cfg, params = _config_from_vector(base_control, base_params, mean)
        mean = _vector_from_config(control_cfg, params)

        if elites[0][0] > best_score:
            top_score, top_vector, top_metrics, top_control, top_params = elites[0]
            best_score = float(top_score)
            best_vector = top_vector.copy()
            best_metrics = dict(top_metrics)
            best_control_cfg = top_control
            best_params_cfg = top_params
            _save_vector(
                outdir / "best_vector.json",
                best_vector,
                best_metrics,
                metadata=_metadata_from_params(best_params_cfg),
            )

        sigma *= args.sigma_decay
        print(
            f"[gen {gen:03d}] score={elites[0][0]:.6f} mse={elites[0][2]['mean_mse']:.6f} "
            f"cycle={elites[0][2]['mean_cycle_error']:.6f} best={best_score:.6f} sigma={sigma:.4f}",
            flush=True,
        )

    final_control, final_params = _config_from_vector(base_control, base_params, best_vector)
    controller = HeatDiffusionController(final_control, final_params.shape)
    simulator = HeatDiffusionArcSimulator(final_params, controller)
    history = simulator.run(steps=args.steps, record_interval=args.record_interval)
    artifact_dir = outdir / "best_artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.save(artifact_dir / "metrics.npy", np.array(history["metrics"], dtype=object))
    vis = simulator.generate_visualization(artifact_dir, history)
    _save_vector(
        outdir / "best_vector.json",
        best_vector,
        best_metrics,
        metadata=_metadata_from_params(final_params),
    )
    print("Training complete. Best results saved in", outdir)
    for key, value in vis.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
