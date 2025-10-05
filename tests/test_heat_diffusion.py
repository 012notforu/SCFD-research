import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.heat_diffusion import (
    HeatDiffusionParams,
    HeatDiffusionControlConfig,
    HeatDiffusionController,
    HeatDiffusionSimulator,
    synthetic_temperature,
)
from run.train_cma_heat import _config_from_vector


def test_synthetic_temperature_bounds():
    pattern = synthetic_temperature((32, 32), kind="hot_corner")
    assert pattern.shape == (32, 32)
    assert np.all(pattern >= 0.0) and np.all(pattern <= 1.0)


def test_heat_controller_shapes():
    cfg = HeatDiffusionControlConfig()
    controller = HeatDiffusionController(cfg, (32, 32))
    err = np.random.default_rng(0).normal(size=(32, 32)).astype(np.float32)
    result = controller.step(err)
    assert result["delta"].shape == (32, 32)


def test_heat_simulator_step():
    params = HeatDiffusionParams(shape=(32, 32), init_seed=1)
    cfg = HeatDiffusionControlConfig(encode_gain=0.2, control_gain=0.002)
    controller = HeatDiffusionController(cfg, params.shape)
    target = synthetic_temperature(params.shape, kind="gradient")
    sim = HeatDiffusionSimulator(params, controller, target)
    stats = sim.step()
    assert np.isfinite(stats["mse"])
    assert np.isfinite(stats["energy"])
    assert np.isfinite(stats["mean_abs_grad"])


@pytest.mark.slow
def test_heat_diffusion_hotcorner_best_vector_regression():
    vector_path = Path("runs/heat_diffusion_cma_hotcorner/best_vector.json")
    data = json.loads(vector_path.read_text())
    vector = np.asarray(data["vector"], dtype=np.float32)

    base_control = HeatDiffusionControlConfig()
    base_params = HeatDiffusionParams()
    control_cfg, params = _config_from_vector(base_control, base_params, vector)

    controller = HeatDiffusionController(control_cfg, params.shape)
    target = synthetic_temperature(params.shape, kind="hot_corner")
    simulator = HeatDiffusionSimulator(params, controller, target)
    history = simulator.run(steps=1200, record_interval=100)
    final_metrics = history["metrics"][-1]

    assert final_metrics["mse"] <= data["metrics"]["max_mse"] * 1.1
    assert np.isfinite(final_metrics["energy"])
    assert final_metrics["energy"] <= data["metrics"]["mean_energy"] * 1.15
