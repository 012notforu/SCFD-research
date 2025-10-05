import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.wave_field import (
    WaveFieldParams,
    WaveFieldControlConfig,
    WaveFieldController,
    WaveFieldSimulator,
    synthetic_wave_target,
)
from run.train_cma_wave import _config_from_vector


def test_wave_target_bounds():
    pattern = synthetic_wave_target((32, 32), kind="focus")
    assert pattern.shape == (32, 32)
    assert np.isfinite(pattern).all()


def test_wave_simulator_step():
    params = WaveFieldParams(shape=(32, 32), dt=0.05)
    control_cfg = WaveFieldControlConfig(control_gain=0.01)
    controller = WaveFieldController(control_cfg, params.shape)
    target = synthetic_wave_target(params.shape, kind="focus")
    sim = WaveFieldSimulator(params, controller, target)
    stats = sim.step()
    assert np.isfinite(stats["mse"])
    assert np.isfinite(stats["energy"])


@pytest.mark.slow
def test_wave_field_defocus_best_vector_regression():
    vector_path = Path("runs/wave_field_cma_defocus/best_vector.json")
    data = json.loads(vector_path.read_text())
    vector = np.asarray(data["vector"], dtype=np.float32)

    base_control = WaveFieldControlConfig()
    base_params = WaveFieldParams()
    control_cfg, params = _config_from_vector(base_control, base_params, vector)

    controller = WaveFieldController(control_cfg, params.shape)
    target = synthetic_wave_target(params.shape, kind="defocus")
    simulator = WaveFieldSimulator(params, controller, target)
    history = simulator.run(steps=1500, record_interval=50)
    final_metrics = history["metrics"][-1]

    assert final_metrics["mse"] <= data["metrics"]["mean_mse"] * 1.1
    assert np.isfinite(final_metrics["energy"])
    assert final_metrics["energy"] <= data["metrics"]["mean_energy"] * 1.2
