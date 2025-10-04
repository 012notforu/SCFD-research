from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from benchmarks.scfd_cartpole import SCFDCartPoleController, SCFDControllerConfig


def _vector_from_config(cfg: SCFDControllerConfig) -> np.ndarray:
    parts = [
        np.asarray(cfg.policy_weights, dtype=np.float32).ravel(),
        np.array([cfg.policy_bias], dtype=np.float32),
        np.array(
            [
                cfg.blend_linear_weight,
                cfg.blend_ternary_weight,
                cfg.ternary_force_scale,
                cfg.ternary_smooth_lambda,
            ],
            dtype=np.float32,
        ),
    ]
    return np.concatenate(parts)


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def _config_from_vector(base: SCFDControllerConfig, vector: np.ndarray) -> SCFDControllerConfig:
    vec = np.asarray(vector, dtype=np.float32)
    idx = 0
    d = base.policy_weights.size
    policy_weights = vec[idx : idx + d]
    idx += d
    policy_bias = float(vec[idx])
    idx += 1
    blend_linear = float(vec[idx])
    idx += 1
    blend_ternary = float(vec[idx])
    idx += 1
    ternary_force = float(vec[idx])
    idx += 1
    ternary_smooth = float(vec[idx])

    weights = np.clip(policy_weights, -8.0, 8.0).astype(np.float32)
    bias = _clip(policy_bias, -5.0, 5.0)
    blend_linear = _clip(blend_linear, 0.0, 2.0)
    blend_ternary = _clip(blend_ternary, 0.0, 2.0)
    ternary_force = _clip(ternary_force, 2.0, 15.0)
    ternary_smooth = _clip(ternary_smooth, 0.05, 1.0)

    return replace(
        base,
        policy_weights=weights,
        policy_bias=bias,
        blend_linear_weight=blend_linear,
        blend_ternary_weight=blend_ternary,
        ternary_force_scale=ternary_force,
        ternary_smooth_lambda=ternary_smooth,
    )


def _evaluate(
    cfg: SCFDControllerConfig,
    seeds: Iterable[int],
    steps: int,
    mass_override: tuple[float, float] | None = None,
) -> Tuple[float, Dict[str, float]]:
    rewards: List[float] = []
    for seed in seeds:
        controller = SCFDCartPoleController(cfg, rng=np.random.default_rng(int(seed)))
        if mass_override is not None:
            masscart, masspole = mass_override
            if masscart is not None:
                controller.physics.masscart = float(masscart)
            if masspole is not None:
                controller.physics.masspole = float(masspole)
        result = controller.run_episode(steps=steps)
        rewards.append(float(result["steps"]))
    arr = np.asarray(rewards, dtype=np.float32)
    metrics = {
        "mean_steps": float(arr.mean()),
        "std_steps": float(arr.std()),
        "max_steps": float(arr.max()),
        "min_steps": float(arr.min()),
    }
    score = metrics["mean_steps"]
    return score, metrics


def _write_history_header(path: Path) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "generation",
                "candidate",
                "score",
                "mean_steps",
                "std_steps",
                "max_steps",
                "min_steps",
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
                metrics["mean_steps"],
                metrics["std_steps"],
                metrics["max_steps"],
                metrics["min_steps"],
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
    parser = argparse.ArgumentParser(description="Tune SCFD blended cart-pole controller with evolutionary search")
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--elite", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=0.5, help="Initial sampling stddev")
    parser.add_argument("--sigma-decay", type=float, default=1.0, help="Multiplicative sigma decay per generation")
    parser.add_argument("--episodes", type=int, default=3, help="Evaluation rollouts per candidate")
    parser.add_argument("--steps", type=int, default=2500, help="Steps per rollout episode")
    parser.add_argument("--masscart", type=float, default=None, help="Override masscart if provided")
    parser.add_argument("--masspole", type=float, default=None, help="Override masspole if provided")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="runs/scfd_cma")
    parser.add_argument("--scfd-cfg", type=str, default="cfg/defaults.yaml", help="SCFD YAML config (for future use)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.elite <= 0 or args.elite > args.population:
        raise ValueError("elite must be in [1, population]")
    mass_override = (args.masscart, args.masspole)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    history_path = outdir / "history.csv"
    _write_history_header(history_path)

    rng = np.random.default_rng(args.seed)
    base_cfg = SCFDControllerConfig()
    mean = _vector_from_config(base_cfg)
    sigma = float(args.sigma)

    seeds = rng.integers(0, 2**32 - 1, size=args.episodes, endpoint=False)

    best_score = -np.inf
    best_vector = mean.copy()
    best_metrics: Dict[str, float] = {}

    dim = mean.size
    for gen in range(args.generations):
        candidates: List[Tuple[float, np.ndarray, Dict[str, float]]] = []

        # Evaluate current mean first
        cfg = _config_from_vector(base_cfg, mean)
        score, metrics = _evaluate(cfg, seeds, args.steps, mass_override=mass_override)
        candidates.append((score, mean.copy(), metrics))
        _append_history(history_path, gen, 0, score, metrics, score > best_score)

        for cand_idx in range(1, args.population):
            sample = mean + sigma * rng.normal(size=dim)
            cfg = _config_from_vector(base_cfg, sample)
            score, metrics = _evaluate(cfg, seeds, args.steps, mass_override=mass_override)
            candidates.append((score, sample, metrics))
            _append_history(history_path, gen, cand_idx, score, metrics, score > best_score)

        candidates.sort(key=lambda item: item[0], reverse=True)
        elites = candidates[: args.elite]

        elite_vectors = np.stack([vec for _, vec, _ in elites], axis=0)
        mean = elite_vectors.mean(axis=0)
        # project mean back into feasible range
        mean = _vector_from_config(_config_from_vector(base_cfg, mean))[...,]

        if elites[0][0] > best_score:
            best_score, best_vector, best_metrics = elites[0]
            _save_vector(outdir / "best_vector.json", best_vector, best_metrics)

        sigma *= args.sigma_decay
        print(
            f"[gen {gen:03d}] best={elites[0][0]:.1f} mean_steps, global_best={best_score:.1f}, sigma={sigma:.3f}",
            flush=True,
        )

    # Final evaluation on best vector for record
    final_cfg = _config_from_vector(base_cfg, best_vector)
    final_score, final_metrics = _evaluate(final_cfg, seeds, args.steps, mass_override=mass_override)
    _save_vector(outdir / "best_vector.json", best_vector, final_metrics)
    print("Training complete. Best mean steps:", final_score)


if __name__ == "__main__":
    main()
