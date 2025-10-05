"""Head-to-head cart-pole benchmark for EM and SCFD controllers."""
from __future__ import annotations

import argparse
import time
from typing import Dict, Tuple
from pathlib import Path

import numpy as np

from .em_cartpole import EMCartPoleController, EMCartConfig
from .scfd_cartpole import SCFDCartPoleController, SCFDControllerConfig


def run_em(args: argparse.Namespace) -> Tuple[Dict[str, float], EMCartPoleController]:
    cfg = EMCartConfig(
        B=args.em_population,
        n_control_steps=args.steps,
        GA_interval=args.ga_interval,
    )
    controller = EMCartPoleController(cfg)
    start = time.perf_counter()
    metrics = controller.run(steps=args.steps)
    metrics["elapsed_sec"] = time.perf_counter() - start
    return metrics, controller


def run_scfd(args: argparse.Namespace) -> Tuple[Dict[str, float], SCFDCartPoleController]:
    micro_calm = args.scfd_micro_steps_calm if args.scfd_micro_steps_calm is not None else max(4, args.scfd_micro_steps // 2)
    cfg = SCFDControllerConfig(
        scfd_cfg_path=args.scfd_cfg,
        micro_steps=args.scfd_micro_steps,
        micro_steps_calm=micro_calm,
        encode_gain=args.scfd_encode_gain,
        encode_width=args.scfd_encode_width,
        decay=args.scfd_decay,
        smooth_lambda=args.scfd_smooth_lambda,
        deadzone_angle=np.deg2rad(args.scfd_deadzone_angle),
        deadzone_ang_vel=args.scfd_deadzone_ang_vel,
        action_delta_clip=args.scfd_action_delta,
    )
    if args.scfd_weights:
        weights = np.array([float(w) for w in args.scfd_weights.split(',') if w.strip()], dtype=np.float32)
        if weights.size != cfg.feature_weights.size:
            raise ValueError(f"Expected {cfg.feature_weights.size} SCFD weights, got {weights.size}")
        cfg.feature_weights = weights
    cfg.gain_energy = args.scfd_gain_energy
    cfg.gain_angle = args.scfd_gain_angle
    cfg.gain_ang_vel = args.scfd_gain_ang_vel
    controller = SCFDCartPoleController(cfg)
    results = []
    start = time.perf_counter()
    for _ in range(args.episodes):
        results.append(controller.run_episode(steps=args.steps))
    elapsed = time.perf_counter() - start
    steps = np.array([r["steps"] for r in results], dtype=np.float32)
    metrics = {
        "mean_steps": float(steps.mean()),
        "std_steps": float(steps.std()),
        "max_steps": float(steps.max()),
        "elapsed_sec": elapsed,
    }
    return metrics, controller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cart-pole benchmark for EM and SCFD controllers")
    parser.add_argument("--controller", choices=["em", "scfd", "both"], default="both")
    parser.add_argument("--steps", type=int, default=20000, help="Number of control steps per evaluation")
    parser.add_argument("--episodes", type=int, default=5, help="SCFD episodes to average over")
    parser.add_argument("--em-population", type=int, default=256, dest="em_population")
    parser.add_argument("--ga-interval", type=int, default=10, dest="ga_interval")
    parser.add_argument("--scfd-cfg", default="cfg/defaults.yaml")
    parser.add_argument("--scfd-micro-steps", type=int, default=40)
    parser.add_argument("--scfd-encode-gain", type=float, default=0.05)
    parser.add_argument("--scfd-encode-width", type=int, default=3)
    parser.add_argument("--scfd-decay", type=float, default=0.98)
    parser.add_argument("--scfd-smooth-lambda", type=float, default=0.25)
    parser.add_argument("--scfd-deadzone-angle", type=float, default=1.0, help="Deadzone angle in degrees")
    parser.add_argument("--scfd-deadzone-ang-vel", type=float, default=0.1)
    parser.add_argument("--scfd-action-delta", type=float, default=2.0)
    parser.add_argument("--scfd-micro-steps-calm", type=int, metavar="MICRO", default=None)
    parser.add_argument("--scfd-weights", type=str, default=None, help="Comma-separated weights for feature policy")
    parser.add_argument("--scfd-gain-energy", type=float, default=4.0)
    parser.add_argument("--scfd-gain-angle", type=float, default=6.0)
    parser.add_argument("--scfd-gain-ang-vel", type=float, default=2.0)
    parser.add_argument("--viz", choices=["none", "em", "scfd", "both"], default="none")
    parser.add_argument("--viz-steps", type=int, default=800)
    parser.add_argument("--outdir", default="cartpole_outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    em_controller = None
    if args.controller in ("em", "both"):
        em_metrics, em_controller = run_em(args)
        print("EM controller metrics:")
        for k, v in em_metrics.items():
            print(f"  {k}: {v}")

    scfd_controller = None
    if args.controller in ("scfd", "both"):
        scfd_metrics, scfd_controller = run_scfd(args)
        print("SCFD controller metrics:")
        for k, v in scfd_metrics.items():
            print(f"  {k}: {v}")

    if args.viz in ("em", "both") and em_controller is not None:
        viz_dir = out_dir / "em"
        result = em_controller.generate_visualization(horizon=args.viz_steps, out_dir=viz_dir)
        print("EM visualization outputs:")
        for k, v in result.items():
            print(f"  {k}: {v}")

    if args.viz in ("scfd", "both") and scfd_controller is not None:
        viz_dir = out_dir / "scfd"
        result = scfd_controller.generate_visualization(steps=args.viz_steps, out_dir=viz_dir)
        print("SCFD visualization outputs:")
        for k, v in result.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
