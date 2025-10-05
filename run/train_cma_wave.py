from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from benchmarks.wave_field import (
    WaveFieldParams,
    WaveFieldControlConfig,
    WaveFieldController,
    WaveFieldSimulator,
    synthetic_wave_target,
)


def _vector_from_config(control: WaveFieldControlConfig, params: WaveFieldParams) -> np.ndarray:
    return np.array(
        [
            control.encode_gain,
            control.encode_decay,
            control.control_gain,
            control.control_clip,
            control.smooth_lambda,
            control.theta_clip,
            params.wave_speed,
        ],
        dtype=np.float32,
    )


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def _config_from_vector(
    base_control: WaveFieldControlConfig,
    base_params: WaveFieldParams,
    vector: np.ndarray,
) -> Tuple[WaveFieldControlConfig, WaveFieldParams]:
    vec = np.asarray(vector, dtype=np.float32)
    encode_gain = _clip(vec[0], 0.0, 5.0)
    encode_decay = _clip(vec[1], 0.5, 0.999)
    control_gain = _clip(vec[2], 1e-4, 5e-2)
    control_clip = _clip(vec[3], 1e-3, 0.2)
    smooth_lambda = _clip(vec[4], 0.0, 1.0)
    theta_clip = _clip(vec[5], 0.5, 5.0)
    wave_speed = _clip(vec[6], 0.5, 2.0)
    control = replace(
        base_control,
        encode_gain=encode_gain,
        encode_decay=encode_decay,
        control_gain=control_gain,
        control_clip=control_clip,
        smooth_lambda=smooth_lambda,
        theta_clip=theta_clip,
    )
    params = replace(base_params, wave_speed=wave_speed)
    return control, params


def _evaluate(
    control_cfg: WaveFieldControlConfig,
    params: WaveFieldParams,
    seeds: Iterable[int],
    steps: int,
    record_interval: int,
    target_kind: str,
) -> Tuple[float, Dict[str, float]]:
    mses: List[float] = []
    energies: List[float] = []
    for seed in seeds:
        sim_params = replace(params, init_seed=int(seed))
        controller = WaveFieldController(control_cfg, sim_params.shape)
        target = synthetic_wave_target(sim_params.shape, kind=target_kind)
        simulator = WaveFieldSimulator(sim_params, controller, target)
        history = simulator.run(steps=steps, record_interval=record_interval)
        last = history["metrics"][-1]
        mses.append(float(last["mse"]))
        energies.append(float(last["energy"]))
    mse_arr = np.array(mses, dtype=np.float32)
    energy_arr = np.array(energies, dtype=np.float32)
    metrics = {
        "mean_mse": float(mse_arr.mean()),
        "std_mse": float(mse_arr.std()),
        "mean_energy": float(energy_arr.mean()),
    }
    score = -metrics["mean_mse"]
    return score, metrics


def _write_history_header(path: Path) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "generation",
            "candidate",
            "score",
            "mean_mse",
            "std_mse",
            "mean_energy",
            "is_best",
        ])


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
        writer.writerow([
            generation,
            candidate,
            score,
            metrics["mean_mse"],
            metrics["std_mse"],
            metrics["mean_energy"],
            int(is_best),
        ])


def _save_vector(path: Path, vector: np.ndarray, metrics: Dict[str, float]) -> None:
    path.write_text(json.dumps({"vector": vector.tolist(), "metrics": metrics}, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CMA-style search for SCFD wave-field controller")
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--elite", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=0.3)
    parser.add_argument("--sigma-decay", type=float, default=0.95)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--record-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", type=str, default="focus", choices=["focus", "defocus", "waves"])
    parser.add_argument("--outdir", type=str, default="runs/wave_field_cma")
    parser.add_argument("--early-stop-score", type=float, default=-1e-3)
    parser.add_argument("--early-stop-count", type=int, default=5)
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
    base_control = WaveFieldControlConfig()
    base_params = WaveFieldParams()
    mean = _vector_from_config(base_control, base_params)
    sigma = float(args.sigma)

    seeds = rng.integers(0, 2**32 - 1, size=args.episodes, endpoint=False)

    best_score = -np.inf
    best_vector = mean.copy()
    best_metrics: Dict[str, float] = {}

    dim = mean.size
    early_counter = 0

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

        if best_score >= args.early_stop_score:
            early_counter += 1
        else:
            early_counter = 0

        sigma *= args.sigma_decay
        print(
            f"[gen {gen:03d}] best_score={elites[0][0]:.6f} mean_mse={elites[0][2]['mean_mse']:.6f} global_best={-best_score:.6f} mse, sigma={sigma:.4f}",
            flush=True,
        )

        if early_counter >= args.early_stop_count:
            print("Early stopping triggered.")
            break

    final_control, final_params = _config_from_vector(base_control, base_params, best_vector)
    controller = WaveFieldController(final_control, final_params.shape)
    target = synthetic_wave_target(final_params.shape, kind=args.target)
    simulator = WaveFieldSimulator(final_params, controller, target)
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
