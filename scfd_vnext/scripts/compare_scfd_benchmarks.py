"""Compare SCFD baseline vs tuned controllers across benchmarks."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SCFD_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SCFD_ROOT.parent
META_DIR = REPO_ROOT / "meta"
DEFAULT_CARTPOLE_VECTOR = META_DIR / "vectors" / "cartpole" / "2025-09-29_half.json"
DEFAULT_GRAY_VECTOR = META_DIR / "vectors" / "gray_scott" / "2025-09-29_half.json"
DEFAULT_HEAT_VECTOR = META_DIR / "vectors" / "heat_diffusion" / "2025-09-29_large.json"
DEFAULT_SCFCFG_PATH = SCFD_ROOT / "cfg" / "defaults.yaml"

import sys
if str(SCFD_ROOT) not in sys.path:
    sys.path.insert(0, str(SCFD_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from benchmarks.scfd_cartpole import SCFDControllerConfig, SCFDCartPoleController
from benchmarks.gray_scott import (
    GrayScottParams,
    GrayScottControlConfig,
    GrayScottController,
    GrayScottSimulator,
    synthetic_target,
)
from benchmarks.heat_diffusion import (
    HeatDiffusionParams,
    HeatDiffusionControlConfig,
    HeatDiffusionController,
    HeatDiffusionSimulator,
    synthetic_temperature,
)
from run.train_cma_scfd import _config_from_vector as cartpole_cfg_from_vector
from run.train_cma_gray_scott import _config_from_vector as gray_cfg_from_vector
from run.train_cma_heat import _config_from_vector as heat_cfg_from_vector


def _mean_std(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def evaluate_cartpole(cfg: SCFDControllerConfig, steps: int, seeds: List[int]) -> Dict[str, float]:
    metrics = []
    for seed in seeds:
        ctrl = SCFDCartPoleController(cfg, rng=np.random.default_rng(seed))
        metrics.append(ctrl.run_episode(steps=steps))
    return {
        "steps": _mean_std([m["steps"] for m in metrics]),
        "rms_step": _mean_std([m["rms_step"] for m in metrics]),
        "action_last": _mean_std([m["action_last"] for m in metrics]),
    }


def evaluate_gray_scott(
    control_cfg: GrayScottControlConfig,
    params: GrayScottParams,
    target_kind: str,
    steps: int,
    record_interval: int,
    seeds: List[int],
) -> Dict[str, float]:
    mses = []
    energies = []
    for seed in seeds:
        sim_params = replace(params, init_seed=seed) if seed != params.init_seed else params
        controller = GrayScottController(control_cfg, sim_params.shape)
        target = synthetic_target(sim_params.shape, kind=target_kind)
        simulator = GrayScottSimulator(sim_params, controller, target)
        history = simulator.run(steps=steps, record_interval=record_interval)
        last = history["metrics"][-1]
        mses.append(float(last["mse"]))
        energies.append(float(last["energy"]))
    return {
        "mse": _mean_std(mses),
        "energy": _mean_std(energies),
    }


def evaluate_heat(
    control_cfg: HeatDiffusionControlConfig,
    params: HeatDiffusionParams,
    target_kind: str,
    steps: int,
    record_interval: int,
    seeds: List[int],
) -> Dict[str, float]:
    mses = []
    energies = []
    grads = []
    for seed in seeds:
        sim_params = replace(params, init_seed=seed) if seed != params.init_seed else params
        controller = HeatDiffusionController(control_cfg, sim_params.shape)
        target = synthetic_temperature(sim_params.shape, kind=target_kind)
        simulator = HeatDiffusionSimulator(sim_params, controller, target)
        history = simulator.run(steps=steps, record_interval=record_interval)
        last = history["metrics"][-1]
        mses.append(float(last["mse"]))
        energies.append(float(last["energy"]))
        grads.append(float(last["mean_abs_grad"]))
    return {
        "mse": _mean_std(mses),
        "energy": _mean_std(energies),
        "mean_abs_grad": _mean_std(grads),
    }


def load_vector(path: Path) -> np.ndarray:
    data = json.loads(path.read_text())
    return np.array(data["vector"], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SCFD baseline vs tuned controllers")
    parser.add_argument("--cartpole-vector", type=Path, default=DEFAULT_CARTPOLE_VECTOR)
    parser.add_argument("--gray-vector", type=Path, default=DEFAULT_GRAY_VECTOR)
    parser.add_argument("--heat-vector", type=Path, default=DEFAULT_HEAT_VECTOR)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "comparisons" / "scfd_vs_tuned.json")
    parser.add_argument("--cartpole-steps", type=int, default=2500)
    parser.add_argument("--gray-steps", type=int, default=1200)
    parser.add_argument("--heat-steps", type=int, default=1500)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    results = {"cartpole": {}, "gray_scott": {}, "heat_diffusion": {}}

    # Cart-pole baseline
    baseline_cfg = SCFDControllerConfig()
    baseline_cfg.scfd_cfg_path = str(DEFAULT_SCFCFG_PATH)
    results["cartpole"]["baseline"] = evaluate_cartpole(baseline_cfg, steps=args.cartpole_steps, seeds=args.seeds)

    tuned_vec = load_vector(args.cartpole_vector)
    tuned_cfg = cartpole_cfg_from_vector(SCFDControllerConfig(), tuned_vec)
    tuned_cfg.scfd_cfg_path = str(DEFAULT_SCFCFG_PATH)
    results["cartpole"]["tuned"] = evaluate_cartpole(tuned_cfg, steps=args.cartpole_steps, seeds=args.seeds)

    # Gray-Scott baseline & tuned
    gray_params = GrayScottParams()
    gray_control = GrayScottControlConfig()
    gray_control = replace(gray_control, scfd_cfg_path=str(DEFAULT_SCFCFG_PATH))
    results["gray_scott"]["baseline"] = evaluate_gray_scott(
        gray_control,
        gray_params,
        target_kind="spots",
        steps=args.gray_steps,
        record_interval=100,
        seeds=args.seeds,
    )

    gray_vec = load_vector(args.gray_vector)
    tuned_control, tuned_params = gray_cfg_from_vector(gray_control, gray_params, gray_vec)
    results["gray_scott"]["tuned"] = evaluate_gray_scott(
        tuned_control,
        tuned_params,
        target_kind="spots",
        steps=args.gray_steps,
        record_interval=100,
        seeds=args.seeds,
    )

    # Heat diffusion baseline & tuned
    heat_params = HeatDiffusionParams()
    heat_control = HeatDiffusionControlConfig()
    heat_control = replace(heat_control, scfd_cfg_path=str(DEFAULT_SCFCFG_PATH))
    results["heat_diffusion"]["baseline"] = evaluate_heat(
        heat_control,
        heat_params,
        target_kind="gradient",
        steps=args.heat_steps,
        record_interval=100,
        seeds=args.seeds,
    )

    heat_vec = load_vector(args.heat_vector)
    tuned_heat_control, tuned_heat_params = heat_cfg_from_vector(heat_control, heat_params, heat_vec)
    results["heat_diffusion"]["tuned"] = evaluate_heat(
        tuned_heat_control,
        tuned_heat_params,
        target_kind="gradient",
        steps=args.heat_steps,
        record_interval=100,
        seeds=args.seeds,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
