import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.heat_diffusion import HeatDiffusionControlConfig, HeatDiffusionController
from benchmarks.heat_diffusion_inverse import (
    HeatInverseParams,
    HeatInverseSimulator,
    synthetic_source_map,
)
from run.train_cma_heat_inverse import _config_from_vector


def test_synthetic_source_map_bounds():
    rng = np.random.default_rng(0)
    field = synthetic_source_map((32, 32), rng=rng)
    assert field.shape == (32, 32)
    assert np.max(np.abs(field)) <= 1.0 + 1e-6


def test_heat_inverse_step_metrics():
    params = HeatInverseParams(shape=(24, 24), forward_steps=6, control_budget=1.0)
    cfg = HeatDiffusionControlConfig(control_gain=0.001, control_clip=0.05)
    controller = HeatDiffusionController(cfg, params.shape)
    sim = HeatInverseSimulator(params, controller)
    baseline = float(np.mean((sim._forward_diffuse(sim.estimate) - sim.observed) ** 2))
    stats = sim.step()
    assert {"obs_mse", "source_mse", "budget_util", "energy"}.issubset(stats)
    assert stats["obs_mse"] <= baseline * 1.05
    assert 0.0 <= stats["budget_util"] <= 1.0



@pytest.mark.slow
def test_heat_inverse_best_vector_regression():
    vector_path = Path('runs/heat_inverse_cma/best_vector.json')
    data = json.loads(vector_path.read_text())
    vector = np.asarray(data['vector'], dtype=np.float32)

    base_control = HeatDiffusionControlConfig()
    base_params = HeatInverseParams()
    control_cfg, params = _config_from_vector(base_control, base_params, vector)

    controller = HeatDiffusionController(control_cfg, params.shape)
    simulator = HeatInverseSimulator(params, controller)
    history = simulator.run(steps=200, record_interval=20)
    final_metrics = history['metrics'][-1]

    allowed_source = data['metrics']['mean_source_mse'] * 1.5 + 5e-4
    allowed_obs = data['metrics']['mean_obs_mse'] * 1.3 + 5e-3
    allowed_energy = data['metrics']['mean_energy'] * 1.3 + 5e-3
    allowed_budget = max(1.0, data['metrics']['mean_budget'] * 1.5 + 0.05)

    assert final_metrics['source_mse'] <= allowed_source
    assert final_metrics['obs_mse'] <= allowed_obs
    assert np.isfinite(final_metrics['energy'])
    assert final_metrics['energy'] <= allowed_energy
    assert final_metrics['budget_util'] <= allowed_budget
