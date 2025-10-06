import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.heat_diffusion import HeatDiffusionControlConfig, HeatDiffusionController, HeatDiffusionParams, HeatDiffusionSimulator, synthetic_temperature
from benchmarks.heat_diffusion_arc import HeatDiffusionArcParams, HeatDiffusionArcSimulator
from orchestrator.pipeline import (
    Plan,
    VectorEntry,
    build_plan,
    load_vector_registry,
    plan_for_environment,
    probe_environment,
)


def _make_heat_simulator(kind: str = "hot_corner") -> HeatDiffusionSimulator:
    params = HeatDiffusionParams(shape=(24, 24))
    controller = HeatDiffusionController(HeatDiffusionControlConfig(), params.shape)
    simulator = HeatDiffusionSimulator(params, controller, synthetic_temperature(params.shape, kind=kind))
    return simulator


def _make_heat_arc_simulator() -> HeatDiffusionArcSimulator:
    params = HeatDiffusionArcParams(shape=(24, 24), transform_cycle=("identity", "rotate90"), transform_cycle_interval=5)
    controller = HeatDiffusionController(HeatDiffusionControlConfig(), params.shape)
    return HeatDiffusionArcSimulator(params, controller)


def test_probe_environment_identifies_hot_corner() -> None:
    def factory() -> HeatDiffusionSimulator:
        return _make_heat_simulator("hot_corner")

    report = probe_environment(factory, steps=20, record_interval=5)
    assert report.physics == "heat"
    assert report.target_kind == "hot_corner"
    assert report.transform_cycle_length == 0


def test_probe_environment_identifies_arc_cycle() -> None:
    def factory() -> HeatDiffusionArcSimulator:
        return _make_heat_arc_simulator()

    report = probe_environment(factory, steps=12, record_interval=4)
    assert report.physics == "heat"
    assert report.target_kind == "arc_cycle"
    assert report.transform_cycle_length == 2


def test_load_vector_registry(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    hot_dir = runs / "heat_diffusion_cma_hotcorner"
    hot_dir.mkdir(parents=True)
    (hot_dir / "best_vector.json").write_text(json.dumps({"vector": [0.0], "metrics": {"mean_mse": 0.01}}))

    arc_dir = runs / "heat_arc_cma"
    arc_dir.mkdir()
    (arc_dir / "best_vector.json").write_text(
        json.dumps(
            {
                "vector": [0.0],
                "metrics": {"mean_mse": 0.02},
                "transform_cycle": ["identity", "rotate90"],
                "transform_cycle_interval": 120,
                "base_target_kind": "gradient",
            }
        )
    )

    entries = load_vector_registry(runs)
    assert {e.vector_id for e in entries} == {"heat_diffusion_cma_hotcorner", "heat_arc_cma"}
    arc_entry = next(e for e in entries if e.vector_id == "heat_arc_cma")
    assert arc_entry.objective == "arc_cycle"
    assert arc_entry.metadata.get("transform_cycle") == ("identity", "rotate90")
    assert arc_entry.metadata.get("transform_cycle_interval") == 120


def test_load_vector_registry_routing_metadata(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    routing_dir = runs / "heat_routing_cma"
    routing_dir.mkdir(parents=True)
    payload = {
        "vector": [0.0],
        "metrics": {"mean_mse": 0.05},
        "initial_centers": [[0.3, 0.3], [0.7, 0.7]],
        "target_centers": [[0.7, 0.3], [0.3, 0.7]],
        "blob_sigma": 0.025,
        "collision_radius": 0.08,
    }
    (routing_dir / "best_vector.json").write_text(json.dumps(payload))

    entries = load_vector_registry(runs)
    routing_entry = next(e for e in entries if e.vector_id == "heat_routing_cma")
    assert routing_entry.metadata.get("initial_center_count") == 2
    assert routing_entry.metadata.get("collision_radius") == pytest.approx(0.08)


def test_build_plan_prefers_arc_cycle() -> None:
    arc_entry = VectorEntry(
        vector_id="heat_arc_cma",
        path=Path("runs/heat_arc_cma/best_vector.json"),
        physics="heat",
        objective="arc_cycle",
        tags=("heat", "arc_cycle"),
    )
    gradient_entry = VectorEntry(
        vector_id="heat_diffusion_cma",
        path=Path("runs/heat_diffusion_cma/best_vector.json"),
        physics="heat",
        objective="gradient",
        tags=("heat", "gradient"),
    )

    def factory() -> HeatDiffusionArcSimulator:
        return _make_heat_arc_simulator()

    plan = plan_for_environment(factory, registry=[arc_entry, gradient_entry], steps=12, record_interval=4)
    assert isinstance(plan, Plan)
    assert plan.steps[0].vector.vector_id == "heat_arc_cma"


def test_build_plan_prefers_matching_arc_cycle_metadata() -> None:
    arc_entry_match = VectorEntry(
        vector_id="heat_arc_match",
        path=Path("runs/heat_arc_match/best_vector.json"),
        physics="heat",
        objective="arc_cycle",
        tags=("heat", "arc_cycle"),
        metadata={
            "metrics": {},
            "transform_cycle": ("identity", "rotate90"),
            "transform_cycle_length": 2,
        },
    )
    arc_entry_other = VectorEntry(
        vector_id="heat_arc_other",
        path=Path("runs/heat_arc_other/best_vector.json"),
        physics="heat",
        objective="arc_cycle",
        tags=("heat", "arc_cycle"),
        metadata={
            "metrics": {},
            "transform_cycle": ("identity", "flip_horizontal"),
            "transform_cycle_length": 2,
        },
    )

    def factory() -> HeatDiffusionArcSimulator:
        return _make_heat_arc_simulator()

    plan = plan_for_environment(
        factory, registry=[arc_entry_other, arc_entry_match], steps=12, record_interval=4
    )
    assert plan.steps[0].vector.vector_id == "heat_arc_match"


def test_plan_for_environment_hot_corner_prefers_matching_vector() -> None:
    hot_entry = VectorEntry(
        vector_id="heat_diffusion_cma_hotcorner",
        path=Path("runs/heat_diffusion_cma_hotcorner/best_vector.json"),
        physics="heat",
        objective="hot_corner",
        tags=("heat", "hot_corner"),
    )
    gradient_entry = VectorEntry(
        vector_id="heat_diffusion_cma",
        path=Path("runs/heat_diffusion_cma/best_vector.json"),
        physics="heat",
        objective="gradient",
        tags=("heat", "gradient"),
    )

    def factory() -> HeatDiffusionSimulator:
        return _make_heat_simulator("hot_corner")

    plan = plan_for_environment(factory, registry=[gradient_entry, hot_entry], steps=20, record_interval=5)
    assert plan.steps[0].vector.vector_id == "heat_diffusion_cma_hotcorner"

def test_load_vector_registry_cartpole_metadata(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    cart_dir = runs / "cartpole_cma"
    cart_dir.mkdir(parents=True)
    payload = {
        "vector": [0.0],
        "metrics": {"mean_steps": 4000.0},
        "controller_config": {
            "scfd_cfg_path": "cfg/defaults.yaml",
            "micro_steps": 40,
            "micro_steps_calm": 16,
            "encode_gain": 0.05,
            "encode_width": 3,
            "decay": 0.98,
            "smooth_lambda": 0.25,
            "deadzone_angle": 0.01,
            "deadzone_ang_vel": 0.1,
            "action_clip": 10.0,
            "action_delta_clip": 2.0,
            "reset_noise": [0.05, 0.05, 0.02, 0.05],
            "feature_momentum": 0.05,
            "deadzone_feature_scale": [0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5],
            "policy_weights": [4.0, 1.5, 0.8, 0.0, 6.0, 2.0, 0.5, 0.2, -0.1, 0.0],
            "policy_bias": 0.0,
            "blend_linear_weight": 1.0,
            "blend_ternary_weight": 0.5,
            "ternary_force_scale": 7.5,
            "ternary_smooth_lambda": 1.0
        },
        "training": {"episodes": 3, "steps": 2500},
        "task": "cartpole_balance"
    }
    (cart_dir / "best_vector.json").write_text(json.dumps(payload))

    entries = load_vector_registry(runs)
    entry = next(e for e in entries if e.vector_id == "cartpole_cma")
    assert entry.physics == "cartpole"
    assert entry.objective == "balance"
    controller_meta = entry.metadata.get("controller_config", {})
    assert controller_meta.get("micro_steps") == 40
    assert entry.metadata.get("task") == "cartpole_balance"
