from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from benchmarks.heat_diffusion import HeatDiffusionControlConfig, HeatDiffusionController
from benchmarks.heat_front_tracking import HeatFrontParams, HeatFrontTrackingSimulator
from run.train_cma_heat_front import _config_from_vector as vector_to_cfg


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run heat front tracking benchmark")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--record-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vector", type=str, default="runs/heat_front_cma/best_vector.json")
    parser.add_argument("--front-radius", type=float, default=None)
    parser.add_argument("--front-width", type=float, default=None)
    parser.add_argument("--outdir", type=str, default="heat_front_outputs")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    vector_path = Path(args.vector)
    payload = json.loads(vector_path.read_text())
    vector = np.asarray(payload["vector"], dtype=np.float32)

    defaults = HeatFrontParams()
    metadata_shape = payload.get("shape")
    shape = tuple(int(x) for x in (metadata_shape or defaults.shape))
    alpha = float(payload.get("alpha", defaults.alpha))
    dt = float(payload.get("dt", defaults.dt))
    noise = float(payload.get("noise", defaults.noise))
    radius = args.front_radius if args.front_radius is not None else float(payload.get("front_radius", defaults.front_radius))
    width = args.front_width if args.front_width is not None else float(payload.get("front_width", defaults.front_width))

    base_control = HeatDiffusionControlConfig()
    base_params = HeatFrontParams(
        shape=shape,
        alpha=alpha,
        dt=dt,
        noise=noise,
        front_radius=radius,
        front_width=width,
    )
    control_cfg, params = vector_to_cfg(base_control, base_params, vector)
    params = replace(params, init_seed=args.seed)
    controller = HeatDiffusionController(control_cfg, params.shape)
    simulator = HeatFrontTrackingSimulator(params, controller)

    history = simulator.run(steps=args.steps, record_interval=args.record_interval)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "metrics.npy", np.array(history["metrics"], dtype=object))
    vis = simulator.generate_visualization(outdir, history)
    print("Heat front metrics (last entry):")
    print(history["metrics"][-1])
    print("Artifacts saved to:")
    for key, value in vis.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
