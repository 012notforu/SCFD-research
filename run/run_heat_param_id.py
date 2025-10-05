from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from benchmarks.heat_diffusion import HeatDiffusionControlConfig, HeatDiffusionController
from benchmarks.heat_parameter_id import HeatParameterIDParams, HeatParameterIDSimulator
from run.train_cma_heat_param_id import _config_from_vector as vector_to_cfg


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run heat parameter identification benchmark")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--record-interval", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vector", type=str, default="runs/heat_param_id_cma/best_vector.json")
    parser.add_argument("--alpha-low", type=float, default=None)
    parser.add_argument("--alpha-high", type=float, default=None)
    parser.add_argument("--split-axis", type=str, default=None, choices=["vertical", "horizontal"])
    parser.add_argument("--outdir", type=str, default="heat_param_id_outputs")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    vector_path = Path(args.vector)
    payload = json.loads(vector_path.read_text())
    vector = np.asarray(payload["vector"], dtype=np.float32)

    defaults = HeatParameterIDParams()
    metadata_shape = payload.get("shape")
    shape = tuple(int(x) for x in (metadata_shape or defaults.shape))
    alpha = float(payload.get("alpha", defaults.alpha))
    dt = float(payload.get("dt", defaults.dt))
    noise = float(payload.get("noise", defaults.noise))
    alpha_low = args.alpha_low if args.alpha_low is not None else float(payload.get("alpha_low", defaults.alpha_low))
    alpha_high = args.alpha_high if args.alpha_high is not None else float(payload.get("alpha_high", defaults.alpha_high))
    split_axis = args.split_axis or payload.get("split_axis", defaults.split_axis)

    base_control = HeatDiffusionControlConfig()
    base_params = HeatParameterIDParams(
        shape=shape,
        alpha=alpha,
        dt=dt,
        noise=noise,
        alpha_low=alpha_low,
        alpha_high=alpha_high,
        split_axis=split_axis,
    )
    control_cfg, params = vector_to_cfg(base_control, base_params, vector)
    params = replace(params, init_seed=args.seed)
    controller = HeatDiffusionController(control_cfg, params.shape)
    simulator = HeatParameterIDSimulator(params, controller)

    history = simulator.run(steps=args.steps, record_interval=args.record_interval)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "metrics.npy", np.array(history["metrics"], dtype=object))
    vis = simulator.generate_visualization(outdir, history)
    print("Heat parameter ID metrics (last entry):")
    print(history["metrics"][-1])
    print("Artifacts saved to:")
    for key, value in vis.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
