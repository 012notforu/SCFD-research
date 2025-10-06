from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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


def _metadata_from_config(
    cfg: SCFDControllerConfig,
    *,
    mass_override: Optional[Tuple[Optional[float], Optional[float]]] = None,
    training: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    controller = {
        "scfd_cfg_path": cfg.scfd_cfg_path,
        "micro_steps": int(cfg.micro_steps),
        "micro_steps_calm": int(cfg.micro_steps_calm),
        "encode_gain": float(cfg.encode_gain),
        "encode_width": int(cfg.encode_width),
        "decay": float(cfg.decay),
        "smooth_lambda": float(cfg.smooth_lambda),
        "deadzone_angle": float(cfg.deadzone_angle),
        "deadzone_ang_vel": float(cfg.deadzone_ang_vel),
        "action_clip": float(cfg.action_clip),
        "action_delta_clip": float(cfg.action_delta_clip),
        "reset_noise": np.asarray(cfg.reset_noise, dtype=np.float32).tolist(),
        "feature_momentum": float(cfg.feature_momentum),
        "deadzone_feature_scale": np.asarray(cfg.deadzone_feature_scale, dtype=np.float32).tolist(),
        "policy_weights": np.asarray(cfg.policy_weights, dtype=np.float32).tolist(),
        "policy_bias": float(cfg.policy_bias),
        "blend_linear_weight": float(cfg.blend_linear_weight),
        "blend_ternary_weight": float(cfg.blend_ternary_weight),
        "ternary_force_scale": float(cfg.ternary_force_scale),
        "ternary_smooth_lambda": float(cfg.ternary_smooth_lambda),
    }
    for _extra in (
        ("gain_energy", getattr(cfg, "gain_energy", None)),
        ("gain_angle", getattr(cfg, "gain_angle", None)),
        ("gain_ang_vel", getattr(cfg, "gain_ang_vel", None)),
    ):
        name, value = _extra
        if value is not None:
            controller[name] = float(value)
    metadata: Dict[str, object] = {
        "task": "cartpole_balance",
        "controller_config": controller,
    }
    if mass_override is not None and any(x is not None for x in mass_override):
        metadata["physics_override"] = {
            "masscart": float(mass_override[0]) if mass_override[0] is not None else None,
            "masspole": float(mass_override[1]) if mass_override[1] is not None else None,
        }
    if training:
        metadata["training"] = training
    return metadata


def _config_from_metadata(data: Dict[str, object]) -> SCFDControllerConfig:
    kwargs = dict(data)
    extras = {}
    for key in ("gain_energy", "gain_angle", "gain_ang_vel"):
        if key in kwargs:
            extras[key] = kwargs.pop(key)
    if "policy_weights" in kwargs:
        kwargs["policy_weights"] = np.asarray(kwargs["policy_weights"], dtype=np.float32)
    if "reset_noise" in kwargs:
        kwargs["reset_noise"] = np.asarray(kwargs["reset_noise"], dtype=np.float32)
    if "deadzone_feature_scale" in kwargs:
        kwargs["deadzone_feature_scale"] = np.asarray(kwargs["deadzone_feature_scale"], dtype=np.float32)
    cfg = SCFDControllerConfig(**kwargs)
    for key, value in extras.items():
        setattr(cfg, key, float(value))
    return cfg


def _evaluate(
    cfg: SCFDControllerConfig,
    seeds: Iterable[int],
    steps: int,
    mass_override: Optional[Tuple[Optional[float], Optional[float]]] = None,
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


def _save_vector(
    path: Path,
    vector: np.ndarray,
    metrics: Dict[str, float],
    metadata: Optional[Dict[str, object]] = None,
) -> None:
    payload: Dict[str, object] = {
        "vector": vector.tolist(),
        "metrics": metrics,
    }
    if metadata:
        payload.update(metadata)
    path.write_text(json.dumps(payload, indent=2))


def _training_metadata(args: argparse.Namespace, seeds: Iterable[int]) -> Dict[str, object]:
    seed_list = [int(s) for s in np.asarray(list(seeds), dtype=np.uint64)]
    return {
        "episodes": int(args.episodes),
        "steps": int(args.steps),
        "generations": int(args.generations),
        "population": int(args.population),
        "elite": int(args.elite),
        "sigma_start": float(args.sigma),
        "sigma_decay": float(args.sigma_decay),
        "seed": int(args.seed),
        "evaluation_seeds": seed_list,
    }


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
    parser.add_argument("--outdir", type=str, default="runs/cartpole_cma")
    parser.add_argument("--scfd-cfg", type=str, default="cfg/defaults.yaml", help="SCFD YAML config (for future use)")
    parser.add_argument("--viz-steps", type=int, default=1200, help="Frames used to generate the best-artifact rollout")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.elite <= 0 or args.elite > args.population:
        raise ValueError("elite must be in [1, population]")
    mass_override: Tuple[Optional[float], Optional[float]] = (args.masscart, args.masspole)
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
    best_cfg = base_cfg

    dim = mean.size
    training_meta = _training_metadata(args, seeds)

    for gen in range(args.generations):
        candidates: List[Tuple[float, np.ndarray, Dict[str, float], SCFDControllerConfig]] = []

        cfg = _config_from_vector(base_cfg, mean)
        score, metrics = _evaluate(cfg, seeds, args.steps, mass_override=mass_override)
        candidates.append((score, mean.copy(), metrics, cfg))
        _append_history(history_path, gen, 0, score, metrics, score > best_score)

        for cand_idx in range(1, args.population):
            sample = mean + sigma * rng.normal(size=dim)
            cfg = _config_from_vector(base_cfg, sample)
            score, metrics = _evaluate(cfg, seeds, args.steps, mass_override=mass_override)
            candidates.append((score, sample, metrics, cfg))
            _append_history(history_path, gen, cand_idx, score, metrics, score > best_score)

        candidates.sort(key=lambda item: item[0], reverse=True)
        elites = candidates[: args.elite]

        elite_vectors = np.stack([vec for _, vec, _, _ in elites], axis=0)
        mean = elite_vectors.mean(axis=0)
        mean = _vector_from_config(_config_from_vector(base_cfg, mean))

        if elites[0][0] > best_score:
            top_score, top_vector, top_metrics, top_cfg = elites[0]
            best_score = float(top_score)
            best_vector = top_vector.copy()
            best_metrics = dict(top_metrics)
            best_cfg = top_cfg
            metadata = _metadata_from_config(best_cfg, mass_override=mass_override, training=training_meta)
            _save_vector(outdir / "best_vector.json", best_vector, best_metrics, metadata=metadata)

        sigma *= args.sigma_decay
        print(
            f"[gen {gen:03d}] best={elites[0][0]:.1f} mean_steps, global_best={best_score:.1f}, sigma={sigma:.3f}",
            flush=True,
        )

    final_cfg = best_cfg
    final_score, final_metrics = _evaluate(final_cfg, seeds, args.steps, mass_override=mass_override)
    metadata = _metadata_from_config(final_cfg, mass_override=mass_override, training=training_meta)
    _save_vector(outdir / "best_vector.json", best_vector, final_metrics, metadata=metadata)

    artifact_dir = outdir / "best_artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    controller = SCFDCartPoleController(final_cfg, rng=np.random.default_rng(args.seed))
    vis = controller.generate_visualization(
        steps=min(args.viz_steps, args.steps),
        out_dir=artifact_dir,
        save_video=False,
        video_format="gif",
    )
    np.save(artifact_dir / "metrics.npy", np.array([final_metrics], dtype=object))
    with (artifact_dir / "metadata.json").open("w") as fh:
        json.dump({"training": training_meta, "visualization": vis}, fh, indent=2)

    print("Training complete. Best mean steps:", final_score)


if __name__ == "__main__":
    main()
