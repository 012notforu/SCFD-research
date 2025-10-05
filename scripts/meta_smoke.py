"""Meta-task smoke loader: evaluate all tuned controllers."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SCFD_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SCFD_ROOT.parent
META_DIR = REPO_ROOT / "meta"
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
    synthetic_target as gray_target,
)
from benchmarks.heat_diffusion import (

    HeatDiffusionParams,

    HeatDiffusionControlConfig,

    HeatDiffusionController,

    HeatDiffusionSimulator,

    synthetic_temperature,

)

from benchmarks.heat_diffusion_anisotropic import (

    HeatAnisotropicParams,

    HeatAnisotropicSimulator,

    synthetic_anisotropic_temperature,

)

from benchmarks.heat_diffusion_periodic import (

    HeatPeriodicParams,

    HeatPeriodicSimulator,

    synthetic_periodic_temperature,

)

from benchmarks.heat_diffusion_obstacle import (

    HeatObstacleParams,

    HeatObstacleSimulator,

    synthetic_obstacle_target,

)

from benchmarks.flow_cylinder import (
    FlowCylinderParams,
    FlowCylinderControlConfig,
    FlowCylinderController,
    FlowCylinderSimulator,
)
from benchmarks.flow_constriction import (
    FlowConstrictionParams,
    FlowConstrictionSimulator,
)
from benchmarks.flow_regime_sweep import (
    FlowRegimeParams,
    FlowRegimeSweep,
)
from benchmarks.wave_field import (
    WaveFieldParams,
    WaveFieldControlConfig,
    WaveFieldController,
    WaveFieldSimulator,
    synthetic_wave_target,
)
from benchmarks.wave_field_cavity import (
    WaveCavityParams,
    WaveCavityControlConfig,
    WaveCavityController,
    WaveCavitySimulator,
    standing_mode_target,
)
from benchmarks.wave_field_partial import (
    WavePartialParams,
    WavePartialControlConfig,
    WavePartialController,
    WavePartialSimulator,
    random_sensor_mask,
)
from run.train_cma_scfd import _config_from_vector as cartpole_from_vec
from run.train_cma_gray_scott import _config_from_vector as gray_from_vec
from run.train_cma_heat import _config_from_vector as heat_from_vec
from run.train_cma_heat_anisotropic import _config_from_vector as heat_aniso_from_vec
from run.train_cma_heat_obstacle import _config_from_vector as heat_obstacle_from_vec
from run.train_cma_flow import _config_from_vector as flow_from_vec
from run.train_cma_flow_constriction import _config_from_vector as flow_constriction_from_vec
from run.train_cma_flow_regime import _config_from_vector as flow_regime_from_vec
from run.train_cma_wave import _config_from_vector as wave_from_vec
from run.train_cma_wave_cavity import _config_from_vector as wave_cavity_from_vec
from run.train_cma_wave_partial import _config_from_vector as wave_partial_from_vec


DOMAIN_VECTOR_DIR = {
    "cartpole_control": "cartpole",
    "reaction_diffusion": "gray_scott",
    "heat_diffusion": "heat_diffusion",
    "flow_control": "flow_cylinder",
    "wave_shaping": "wave_field",
}


def _latest_vector(domain: str) -> Path:
    folder = META_DIR / "vectors" / DOMAIN_VECTOR_DIR[domain]
    vectors = sorted(folder.glob("*.json"))
    if not vectors:
        raise FileNotFoundError(f"No vectors found for domain {domain} in {folder}")
    return vectors[-1]


def _mean(values: List[float]) -> float:
    return float(np.mean(np.array(values, dtype=np.float32)))


def evaluate_cartpole(task: Dict[str, object], vector_path: Path, steps: int, seeds: List[int]) -> Dict[str, float]:
    data = json.loads(vector_path.read_text())
    vec = np.array(data["vector"], dtype=np.float32)
    cfg = cartpole_from_vec(SCFDControllerConfig(scfd_cfg_path=str(DEFAULT_SCFCFG_PATH)), vec)
    results = []
    for seed in seeds:
        ctrl = SCFDCartPoleController(cfg, rng=np.random.default_rng(seed))
        results.append(ctrl.run_episode(steps=steps)["steps"])
    return {"mean_steps": _mean(results)}


def evaluate_gray(task: Dict[str, object], vector_path: Path, steps: int, seeds: List[int]) -> Dict[str, float]:
    data = json.loads(vector_path.read_text())
    vec = np.array(data["vector"], dtype=np.float32)
    scfd_cfg = _resolve_cfg_path(task.get("controller", {}).get("scfd_cfg", DEFAULT_SCFCFG_PATH))
    base_control = GrayScottControlConfig(scfd_cfg_path=scfd_cfg)
    params_cfg = dict(task.get("simulator", {}).get("params", {}))
    regime = task.get("regime", "standard")
    bounds_cfg = task.get("bounds", {})
    f_bounds = tuple(bounds_cfg.get("F", (0.01, 0.08)))
    k_bounds = tuple(bounds_cfg.get("k", (0.02, 0.09)))
    base_params = GrayScottParams(**_filter_params(GrayScottParams, params_cfg))
    if regime == "near_turing":
        if "F" not in params_cfg:
            base_params = replace(base_params, F=0.0315)
        if "k" not in params_cfg:
            base_params = replace(base_params, k=0.059)
        if "F" not in bounds_cfg:
            f_bounds = (0.028, 0.034)
        if "k" not in bounds_cfg:
            k_bounds = (0.057, 0.063)
    control, params = gray_from_vec(base_control, base_params, vec, f_bounds=f_bounds, k_bounds=k_bounds)
    target_kind = task.get("target", {}).get("kind", "spots")
    if regime == "near_turing" and target_kind == "spots":
        target_kind = "hover"
    mses = []
    for seed in seeds:
        sim_params = replace(params, init_seed=seed)
        controller = GrayScottController(control, sim_params.shape)
        target = gray_target(sim_params.shape, kind=target_kind)
        simulator = GrayScottSimulator(sim_params, controller, target)
        hist = simulator.run(steps=steps, record_interval=100)
        mses.append(float(hist["metrics"][-1]["mse"]))
    return {"mean_mse": _mean(mses)}


def evaluate_heat(task: Dict[str, object], vector_path: Path, steps: int, seeds: List[int]) -> Dict[str, float]:
    variant = task.get("variant", "baseline")
    data = json.loads(vector_path.read_text())
    vec = np.array(data["vector"], dtype=np.float32)
    params_cfg = dict(task.get("simulator", {}).get("params", {}))
    target_cfg = dict(task.get("target", {}))
    scfd_cfg = _resolve_cfg_path(task.get("controller", {}).get("scfd_cfg", DEFAULT_SCFCFG_PATH))
    base_control = HeatDiffusionControlConfig(scfd_cfg_path=scfd_cfg)

    if variant == "anisotropic":
        base_params = HeatAnisotropicParams(**_filter_params(HeatAnisotropicParams, params_cfg))
        control, tuned_params = heat_aniso_from_vec(base_control, base_params, vec)
        angle = float(np.deg2rad(target_cfg.get("angle_deg", 0.0)))
        kind = target_cfg.get("kind", "tilted")
        mses, ratios = [], []
        for seed in seeds:
            sim_params = replace(tuned_params, init_seed=int(seed))
            controller = HeatDiffusionController(control, sim_params.shape)
            target = synthetic_anisotropic_temperature(sim_params.shape, kind=kind, angle=angle)
            sim = HeatAnisotropicSimulator(sim_params, controller, target)
            last = sim.run(steps=steps, record_interval=100)["metrics"][-1]
            mses.append(float(last["mse"]))
            ratios.append(float(last["principal_ratio"]))
        return {"mean_mse": _mean(mses), "mean_principal_ratio": _mean(ratios)}

    if variant == "obstacle":
        base_params = HeatObstacleParams(**_filter_params(HeatObstacleParams, params_cfg))
        control, tuned_params = heat_obstacle_from_vec(base_control, base_params, vec)
        target_kind = target_cfg.get("kind", "hot_corner")
        mses, corners = [], []
        for seed in seeds:
            sim_params = replace(tuned_params, init_seed=int(seed))
            controller = HeatDiffusionController(control, sim_params.shape)
            target = synthetic_obstacle_target(sim_params.shape, kind=target_kind)
            sim = HeatObstacleSimulator(sim_params, controller, target)
            last = sim.run(steps=steps, record_interval=100)["metrics"][-1]
            mses.append(float(last["mse"]))
            corners.append(float(last["corner_mse"]))
        return {"mean_mse": _mean(mses), "mean_corner_mse": _mean(corners)}

    is_periodic = variant == "periodic" or "dt_jitter" in params_cfg
    if is_periodic:
        base_params = HeatPeriodicParams(**_filter_params(HeatPeriodicParams, params_cfg))
        control, _ = heat_from_vec(base_control, HeatDiffusionParams(), vec)
        mses = []
        for seed in seeds:
            sim_params = replace(base_params, init_seed=int(seed))
            controller = HeatDiffusionController(control, sim_params.shape)
            target = synthetic_periodic_temperature(
                sim_params.shape,
                kind=target_cfg.get("kind", "stripe"),
                phase=target_cfg.get("phase", 0.0),
            )
            sim = HeatPeriodicSimulator(sim_params, controller, target)
            mses.append(float(sim.run(steps=steps, record_interval=100)["metrics"][-1]["mse"]))
        return {"mean_mse": _mean(mses)}

    base_params = HeatDiffusionParams(**_filter_params(HeatDiffusionParams, params_cfg))
    control, tuned_params = heat_from_vec(base_control, base_params, vec)
    target_kind = target_cfg.get("kind", "gradient")
    mses = []
    for seed in seeds:
        sim_params = replace(tuned_params, init_seed=int(seed))
        controller = HeatDiffusionController(control, sim_params.shape)
        target = synthetic_temperature(sim_params.shape, kind=target_kind)
        sim = HeatDiffusionSimulator(sim_params, controller, target)
        mses.append(float(sim.run(steps=steps, record_interval=100)["metrics"][-1]["mse"]))
    return {"mean_mse": _mean(mses)}


def evaluate_flow(task: Dict[str, object], vector_path: Path, steps: int, seeds: List[int]) -> Dict[str, float]:
    variant = task.get("variant", "baseline")
    data = json.loads(vector_path.read_text())
    vec = np.array(data["vector"], dtype=np.float32)
    scfd_cfg = _resolve_cfg_path(task.get("controller", {}).get("scfd_cfg", DEFAULT_SCFCFG_PATH))
    if variant == "constriction":
        params_cfg = dict(task.get("simulator", {}).get("params", {}))
        base_params = FlowConstrictionParams(**params_cfg)
        control, params = flow_constriction_from_vec(FlowCylinderControlConfig(scfd_cfg_path=scfd_cfg), base_params, vec)
        throughputs = []
        backflows = []
        for seed in seeds:
            sim_params = replace(params, init_seed=seed)
            controller = FlowCylinderController(control, sim_params.shape)
            simulator = FlowConstrictionSimulator(sim_params, controller)
            hist = simulator.run(steps=steps, record_interval=60)
            last = hist["metrics"][-1]
            throughputs.append(float(last["throughput"]))
            backflows.append(float(last["backflow"]))
        return {
            "mean_throughput": _mean(throughputs),
            "mean_backflow": _mean(backflows),
        }
    if variant == "regime_sweep":
        params_cfg = dict(task.get("simulator", {}).get("params", {}))
        base_params = FlowRegimeParams(**params_cfg)
        control_cfg, params = flow_regime_from_vec(FlowCylinderControlConfig(scfd_cfg_path=scfd_cfg), base_params, vec)
        sweep = FlowRegimeSweep(params, control_cfg)
        sweep.run(seeds=seeds)
        metrics = sweep.aggregate_metrics()
        return metrics

    params_cfg = task.get("simulator", {}).get("params", {})
    base_params = FlowCylinderParams(**params_cfg)
    control, params = flow_from_vec(FlowCylinderControlConfig(scfd_cfg_path=scfd_cfg), base_params, vec)
    mses = []
    for seed in seeds:
        sim_params = replace(params, init_seed=seed)
        controller = FlowCylinderController(control, sim_params.shape)
        simulator = FlowCylinderSimulator(sim_params, controller)
        hist = simulator.run(steps=steps, record_interval=40)
        mses.append(float(hist["metrics"][-1]["wake_mse"]))
    return {"mean_wake_mse": _mean(mses)}


def evaluate_wave(task: Dict[str, object], vector_path: Path, steps: int, seeds: List[int]) -> Dict[str, float]:
    variant = task.get("variant", "baseline")
    data = json.loads(vector_path.read_text())
    vec = np.array(data["vector"], dtype=np.float32)
    if variant == "cavity":
        params_cfg = dict(task.get("simulator", {}).get("params", {}))
        base_params = WaveCavityParams(**params_cfg)
        control, params = wave_cavity_from_vec(WaveCavityControlConfig(scfd_cfg_path=str(DEFAULT_SCFCFG_PATH)), base_params, vec)
        mses = []
        boundary = []
        for seed in seeds:
            sim_params = replace(params, init_seed=seed)
            controller = WaveCavityController(control, sim_params.shape)
            target = standing_mode_target(sim_params.shape, sim_params.mode_m, sim_params.mode_n)
            simulator = WaveCavitySimulator(sim_params, controller, target)
            hist = simulator.run(steps=steps, record_interval=50)
            last = hist["metrics"][-1]
            mses.append(float(last["mse"]))
            boundary.append(float(last["boundary_energy"]))
        return {
            "mean_mse": _mean(mses),
            "mean_boundary_energy": _mean(boundary),
        }
    if variant == "partial":
        params_cfg = dict(task.get("simulator", {}).get("params", {}))
        base_params = WavePartialParams(**params_cfg)
        control, params = wave_partial_from_vec(WavePartialControlConfig(scfd_cfg_path=str(DEFAULT_SCFCFG_PATH)), base_params, vec)
        mses = []
        sensed = []
        for seed in seeds:
            sim_params = replace(params, init_seed=seed)
            controller = WavePartialController(control, sim_params.shape, sim_params.action_delay)
            rng = np.random.default_rng(seed)
            mask = random_sensor_mask(sim_params.shape, sim_params.sensor_fraction, rng)
            target = synthetic_wave_target(sim_params.shape, kind=task.get("target", {}).get("kind", "focus"))
            simulator = WavePartialSimulator(sim_params, controller, target, mask)
            hist = simulator.run(steps=steps, record_interval=50)
            last = hist["metrics"][-1]
            mses.append(float(last["mse"]))
            sensed.append(float(last["sensed_mse"]))
        return {
            "mean_mse": _mean(mses),
            "mean_sensed_mse": _mean(sensed),
        }

    params_cfg = task.get("simulator", {}).get("params", {})
    base_params = WaveFieldParams(**params_cfg)
    control, params = wave_from_vec(WaveFieldControlConfig(scfd_cfg_path=str(DEFAULT_SCFCFG_PATH)), base_params, vec)
    mses = []
    for seed in seeds:
        sim_params = replace(params, init_seed=seed)
        controller = WaveFieldController(control, sim_params.shape)
        target = synthetic_wave_target(sim_params.shape, kind=task.get("target", {}).get("kind", "focus"))
        simulator = WaveFieldSimulator(sim_params, controller, target)
        hist = simulator.run(steps=steps, record_interval=50)
        mses.append(float(hist["metrics"][-1]["mse"]))
    return {"mean_mse": _mean(mses)}


DOMAIN_EVAL = {

    "cartpole_control": evaluate_cartpole,

    "reaction_diffusion": evaluate_gray,

    "heat_diffusion": evaluate_heat,

    "flow_control": evaluate_flow,

    "wave_shaping": evaluate_wave,

}





def _resolve_cfg_path(path_str):

    import pathlib

    cfg_path = pathlib.Path(path_str)

    if not cfg_path.is_absolute():

        cfg_path = (SCFD_ROOT / cfg_path).resolve()

    return str(cfg_path)





DEFAULT_STEPS = {

    "cartpole_control": 2500,

    "reaction_diffusion": 1200,

    "heat_diffusion": 1500,

    "flow_control": 1200,

    "wave_shaping": 1500,

}





def _filter_params(cls, params):

    annotations = getattr(cls, "__annotations__", {})

    allowed = set(annotations.keys())

    return {k: v for k, v in params.items() if k in allowed}





def main() -> None:
    parser = argparse.ArgumentParser(description="Load all meta tasks and evaluate tuned controllers")
    parser.add_argument("--tasks-dir", type=Path, default=META_DIR / "tasks")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "comparisons" / "meta_smoke.json")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, float]] = {}

    for task_file in sorted(args.tasks_dir.glob("*.json")):
        task = json.loads(task_file.read_text(encoding='utf-8-sig'))
        domain = task["domain"]
        evaluator = DOMAIN_EVAL.get(domain)
        if evaluator is None:
            print(f"Skipping {task_file.name}: unsupported domain {domain}")
            continue
        vector_path = Path(task.get("vector_path")) if task.get("vector_path") else _latest_vector(domain)
        if not vector_path.is_absolute():
            vector_path = REPO_ROOT / vector_path
        if not vector_path.exists():
            print(f"{task['task']}: vector not found at {vector_path}, skipping")
            continue
        steps = task.get("steps", DEFAULT_STEPS[domain])
        metrics = evaluator(task, vector_path, steps=steps, seeds=args.seeds)
        results[task["task"]] = {
            "vector": str(vector_path.relative_to(REPO_ROOT)),
            **metrics,
        }
        print(f"{task['task']}: {metrics}")

    args.output.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()




