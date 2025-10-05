import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.heat_diffusion import HeatDiffusionControlConfig, HeatDiffusionController
from benchmarks.heat_diffusion_arc import (
    HeatDiffusionArcParams,
    HeatDiffusionArcSimulator,
    apply_arc_transform,
    build_transform_cycle,
)
from run.train_cma_heat_arc import _config_from_vector


@pytest.mark.parametrize(
    "transform",
    [
        "identity",
        "rotate90",
        "rotate180",
        "rotate270",
        "flip_horizontal",
        "flip_vertical",
        "diag",
        "anti_diag",
    ],
)
def test_apply_arc_transform_dimensions(transform: str) -> None:
    field = np.arange(16, dtype=np.float32).reshape(4, 4)
    result = apply_arc_transform(field, transform)
    assert result.shape == field.shape


def test_build_transform_cycle_defaults() -> None:
    cycle = build_transform_cycle(["identity", "rotate90", "rotate90", "flip_vertical"])
    assert cycle == ("identity", "rotate90", "rotate90", "flip_vertical")


def test_build_transform_cycle_invalid() -> None:
    with pytest.raises(ValueError):
        build_transform_cycle(["identity", "unknown_transform"])


@pytest.mark.parametrize("names", [[""], []])
def test_build_transform_cycle_empty(names) -> None:
    cycle = build_transform_cycle(names)
    assert cycle == ("identity",)


def test_heat_arc_simulator_step_metrics() -> None:
    params = HeatDiffusionArcParams(shape=(24, 24), transform_cycle=("identity", "rotate90"), transform_cycle_interval=3)
    controller = HeatDiffusionController(HeatDiffusionControlConfig(control_gain=0.002), params.shape)
    sim = HeatDiffusionArcSimulator(params, controller)
    stats = sim.step()
    assert {"mse", "energy", "cycle_error"}.issubset(stats)
    assert np.isfinite(stats["mse"]) and stats["mse"] >= 0.0


def test_heat_arc_run_records_history() -> None:
    params = HeatDiffusionArcParams(shape=(16, 16), transform_cycle=("identity", "rotate180"), transform_cycle_interval=5)
    controller = HeatDiffusionController(HeatDiffusionControlConfig(), params.shape)
    sim = HeatDiffusionArcSimulator(params, controller)
    history = sim.run(steps=12, record_interval=4)
    assert history["history"].shape[0] >= 3
    assert history["targets"].shape[0] == history["history"].shape[0]


def test_heat_arc_transform_cycle_progression() -> None:
    params = HeatDiffusionArcParams(
        shape=(12, 12),
        transform_cycle=("identity", "rotate90"),
        transform_cycle_interval=1,
    )
    controller = HeatDiffusionController(HeatDiffusionControlConfig(control_gain=0.0015), params.shape)
    sim = HeatDiffusionArcSimulator(params, controller)
    history = sim.run(steps=3, record_interval=1)
    transforms = [entry["transform"] for entry in history["metrics"][:2]]
    assert transforms == ["identity", "rotate90"]


@pytest.mark.slow
@pytest.mark.skipif(
    not Path("runs/heat_arc_cma/best_vector.json").exists(),
    reason="heat_arc best vector not yet trained",
)
def test_heat_arc_best_vector_regression() -> None:
    vector_path = Path("runs/heat_arc_cma/best_vector.json")
    data = json.loads(vector_path.read_text())
    vector = np.asarray(data["vector"], dtype=np.float32)

    base_control = HeatDiffusionControlConfig()
    base_params = HeatDiffusionArcParams()
    control_cfg, params = _config_from_vector(base_control, base_params, vector)
    controller = HeatDiffusionController(control_cfg, params.shape)
    simulator = HeatDiffusionArcSimulator(params, controller)
    history = simulator.run(steps=800, record_interval=100)
    final_metrics = history["metrics"][-1]
    baseline_mse = data["metrics"].get("mean_mse", 1.0)
    baseline_cycle = data["metrics"].get("mean_cycle_error", 1.0)
    assert final_metrics["mse"] <= baseline_mse * 1.2
    assert final_metrics["cycle_error"] <= baseline_cycle * 1.2