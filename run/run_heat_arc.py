from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from benchmarks.heat_diffusion import HeatDiffusionControlConfig, HeatDiffusionController
from benchmarks.heat_diffusion_arc import (
    HeatDiffusionArcParams,
    HeatDiffusionArcSimulator,
    build_transform_cycle,
)
from run.train_cma_heat_arc import _config_from_vector as vector_to_cfg


DEFAULT_TRANSFORM_CYCLE = (
    "identity",
    "rotate90",
    "flip_horizontal",
    "diag",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ARC-style heat diffusion benchmark")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--record-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vector", type=str, default="runs/heat_arc_cma/best_vector.json")
    parser.add_argument("--transforms", type=str, default=None, help="Override transform cycle (comma separated)")
    parser.add_argument("--cycle-interval", type=int, default=None, help="Override transform interval")
    parser.add_argument("--base-kind", type=str, default=None, help="Override base target kind")
    parser.add_argument("--outdir", type=str, default="heat_arc_outputs")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    vector_path = Path(args.vector)
    data = json.loads(vector_path.read_text())
    vector = np.asarray(data["vector"], dtype=np.float32)

    metadata_cycle = data.get("transform_cycle")
    metadata_interval = data.get("transform_cycle_interval")
    metadata_kind = data.get("base_target_kind")
    metadata_shape = data.get("shape")
    metadata_alpha = data.get("alpha")
    metadata_dt = data.get("dt")
    metadata_noise = data.get("noise")

    if args.transforms:
        transform_cycle = build_transform_cycle(args.transforms.split(","))
    elif metadata_cycle:
        transform_cycle = build_transform_cycle(metadata_cycle)
    else:
        transform_cycle = build_transform_cycle(DEFAULT_TRANSFORM_CYCLE)

    if args.cycle_interval is not None:
        cycle_interval = max(1, args.cycle_interval)
    elif metadata_interval is not None:
        cycle_interval = max(1, int(metadata_interval))
    else:
        cycle_interval = 200

    base_kind = args.base_kind or metadata_kind or "gradient"

    defaults = HeatDiffusionArcParams()
    shape = tuple(int(x) for x in (metadata_shape or defaults.shape))
    alpha = float(metadata_alpha) if metadata_alpha is not None else defaults.alpha
    dt = float(metadata_dt) if metadata_dt is not None else defaults.dt
    noise = float(metadata_noise) if metadata_noise is not None else defaults.noise

    base_control = HeatDiffusionControlConfig()
    base_params = HeatDiffusionArcParams(
        shape=shape,
        alpha=alpha,
        dt=dt,
        noise=noise,
        transform_cycle=transform_cycle,
        transform_cycle_interval=cycle_interval,
        base_target_kind=base_kind,
    )
    control_cfg, params = vector_to_cfg(base_control, base_params, vector)
    params = replace(params, init_seed=args.seed)
    controller = HeatDiffusionController(control_cfg, params.shape)
    simulator = HeatDiffusionArcSimulator(params, controller)

    history = simulator.run(steps=args.steps, record_interval=args.record_interval)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "metrics.npy", np.array(history["metrics"], dtype=object))
    vis = simulator.generate_visualization(outdir, history)
    print("Heat ARC metrics (last entry):")
    print(history["metrics"][-1])
    print("Artifacts saved to:")
    for key, value in vis.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()