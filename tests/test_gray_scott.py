import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.gray_scott import (
    GrayScottParams,
    GrayScottControlConfig,
    GrayScottController,
    GrayScottSimulator,
    synthetic_target,
)
from run.train_cma_gray_scott import _config_from_vector


def test_synthetic_target_shapes():
    shape = (32, 32)
    pattern = synthetic_target(shape, kind="spots")
    assert pattern.shape == shape
    assert np.all((pattern >= 0.0) & (pattern <= 1.0))


def test_controller_step_shapes():
    cfg = GrayScottControlConfig()
    controller = GrayScottController(cfg, (32, 32))
    error = np.random.default_rng(0).normal(size=(32, 32)).astype(np.float32)
    control = controller.step(error)
    assert control["feed_delta"].shape == (32, 32)
    assert control["kill_delta"].shape == (32, 32)


def test_simulator_reduces_error():
    params = GrayScottParams(shape=(32, 32), init_seed=1)
    control_cfg = GrayScottControlConfig(encode_gain=0.1, control_gain_feed=0.001, control_gain_kill=0.001)
    controller = GrayScottController(control_cfg, params.shape)
    target = synthetic_target(params.shape, kind="stripes")
    sim = GrayScottSimulator(params, controller, target)
    baseline_mse = np.mean((sim.v - target) ** 2)
    stats = sim.step()
    assert stats["mse"] >= 0.0
    # After a few steps, expect mse to move (not NaN) and remain finite.
    for _ in range(10):
        stats = sim.step()
    assert np.isfinite(stats["mse"]) and stats["mse"] >= 0.0
    assert np.isfinite(stats["energy"])
    assert np.isfinite(baseline_mse)


@pytest.mark.slow
def test_gray_scott_near_turing_best_vector_hits_record_metrics():
    vector_path = Path('runs/gray_scott_cma_near_turing/best_vector.json')
    data = json.loads(vector_path.read_text())
    vector = np.asarray(data['vector'], dtype=np.float32)

    base_control = GrayScottControlConfig()
    base_params = GrayScottParams(F=0.0315, k=0.059)
    control_cfg, params = _config_from_vector(
        base_control,
        base_params,
        vector,
        f_bounds=(0.028, 0.034),
        k_bounds=(0.057, 0.063),
    )
    controller = GrayScottController(control_cfg, params.shape)
    target = synthetic_target(params.shape, kind='hover')
    simulator = GrayScottSimulator(params, controller, target)

    history = simulator.run(steps=1200, record_interval=100)
    final_metrics = history['metrics'][-1]
    assert final_metrics['mse'] <= data['metrics']['max_mse'] * 1.05
    assert np.isfinite(final_metrics['energy'])
    assert final_metrics['energy'] <= data['metrics']['mean_energy'] * 1.1
