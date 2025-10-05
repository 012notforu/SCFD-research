import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.heat_diffusion import HeatDiffusionControlConfig, HeatDiffusionController, synthetic_temperature
from benchmarks.heat_diffusion_mobile import (
    HeatMobileParams,
    HeatMobileSimulator,
    synthetic_mobile_target,
)
from run.train_cma_heat_mobile import _config_from_vector


def test_synthetic_temperature_center_overrides():
    pattern_default = synthetic_temperature((32, 32), kind="hot_corner")
    pattern_center = synthetic_temperature((32, 32), kind="hot_corner", center=(24, 8))
    assert pattern_default.shape == pattern_center.shape == (32, 32)
    assert not np.allclose(pattern_default, pattern_center)


def test_synthetic_mobile_target_stack():
    path = ((8, 8), (8, 24), (24, 24))
    targets = synthetic_mobile_target((32, 32), path, radius=3)
    assert targets.shape == (3, 32, 32)
    assert np.all(targets >= 0.0) and np.all(targets <= 1.0)


def test_heat_mobile_step_metrics():
    params = HeatMobileParams(
        shape=(32, 32),
        path=((8, 8), (16, 24)),
        steps_per_waypoint=5,
        heater_radius=2,
        control_budget=1.0,
    )
    cfg = HeatDiffusionControlConfig(control_gain=0.001, control_clip=0.05)
    controller = HeatDiffusionController(cfg, params.shape)
    sim = HeatMobileSimulator(params, controller)
    stats = sim.step()
    assert {"mse", "energy", "budget_util", "path_progress"}.issubset(stats)
    assert 0.0 <= stats["budget_util"] <= 1.0
    assert stats["path_progress"] >= 0.0


@pytest.mark.slow
def test_heat_mobile_best_vector_regression():
    vector_path = Path('runs/heat_mobile_cma/best_vector.json')
    data = json.loads(vector_path.read_text())
    vector = np.asarray(data['vector'], dtype=np.float32)

    base_control = HeatDiffusionControlConfig()
    base_params = HeatMobileParams()
    control_cfg, params = _config_from_vector(base_control, base_params, vector)

    controller = HeatDiffusionController(control_cfg, params.shape)
    simulator = HeatMobileSimulator(params, controller)
    history = simulator.run(steps=1600, record_interval=80)
    final_metrics = history['metrics'][-1]

    assert final_metrics['mse'] <= data['metrics']['mean_mse'] * 1.2
    assert np.isfinite(final_metrics['energy'])
    assert final_metrics['energy'] <= data['metrics']['mean_energy'] * 1.2
    assert final_metrics['budget_util'] <= max(1.0, data['metrics']['mean_budget'] * 1.5 + 0.05)
    assert final_metrics['path_progress'] >= data['metrics']['mean_progress'] * 0.9
