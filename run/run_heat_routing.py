from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

from benchmarks.heat_diffusion import HeatDiffusionControlConfig, HeatDiffusionController
from benchmarks.heat_diffusion_routing import HeatDiffusionRoutingParams, HeatDiffusionRoutingSimulator
from run.train_cma_heat_routing import _config_from_vector as vector_to_cfg


def _parse_centers(arg: str | None) -> Tuple[Tuple[float, float], ...] | None:
    if arg is None:
        return None
    data = json.loads(arg)
    centers: list[Tuple[float, float]] = []
    for entry in data:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError("centers must be a list of pairs")
        y, x = float(entry[0]), float(entry[1])
        centers.append((y, x))
    return tuple(centers)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run heat diffusion routing benchmark")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--record-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vector", type=str, default="runs/heat_routing_cma/best_vector.json")
    parser.add_argument("--initial-centers", type=str, default=None, help="JSON list of [y, x] pairs")
    parser.add_argument("--target-centers", type=str, default=None, help="JSON list of [y, x] pairs")
    parser.add_argument("--blob-sigma", type=float, default=None)
    parser.add_argument("--collision-radius", type=float, default=None)
    parser.add_argument("--outdir", type=str, default="heat_routing_outputs")
    return parser.parse_args()


def _coerce_centers(raw: Sequence[Sequence[float]] | None) -> Tuple[Tuple[float, float], ...]:
    if not raw:
        return tuple()
    return tuple((float(y), float(x)) for y, x in raw)


def main() -> None:
    args = _parse_args()
    vector_path = Path(args.vector)
    payload = json.loads(vector_path.read_text())
    vector = np.asarray(payload["vector"], dtype=np.float32)

    defaults = HeatDiffusionRoutingParams()
    metadata_shape = payload.get("shape")
    shape = tuple(int(x) for x in (metadata_shape or defaults.shape))
    alpha = float(payload.get("alpha", defaults.alpha))
    dt = float(payload.get("dt", defaults.dt))
    noise = float(payload.get("noise", defaults.noise))

    metadata_initial = _coerce_centers(payload.get("initial_centers"))
    metadata_target = _coerce_centers(payload.get("target_centers"))
    metadata_sigma = payload.get("blob_sigma")
    metadata_collision = payload.get("collision_radius")

    initial_centers = _parse_centers(args.initial_centers) or metadata_initial or defaults.initial_centers
    target_centers = _parse_centers(args.target_centers) or metadata_target or defaults.target_centers
    blob_sigma = args.blob_sigma if args.blob_sigma is not None else float(metadata_sigma or defaults.blob_sigma)
    collision_radius = args.collision_radius if args.collision_radius is not None else float(metadata_collision or defaults.collision_radius)

    base_control = HeatDiffusionControlConfig()
    base_params = HeatDiffusionRoutingParams(
        shape=shape,
        alpha=alpha,
        dt=dt,
        noise=noise,
        initial_centers=initial_centers,
        target_centers=target_centers,
        blob_sigma=blob_sigma,
        collision_radius=collision_radius,
    )
    control_cfg, params = vector_to_cfg(base_control, base_params, vector)
    params = replace(params, init_seed=args.seed)
    controller = HeatDiffusionController(control_cfg, params.shape)
    simulator = HeatDiffusionRoutingSimulator(params, controller)

    history = simulator.run(steps=args.steps, record_interval=args.record_interval)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "metrics.npy", np.array(history["metrics"], dtype=object))
    vis = simulator.generate_visualization(outdir, history)
    print("Heat routing metrics (last entry):")
    print(history["metrics"][-1])
    print("Artifacts saved to:")
    for key, value in vis.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
