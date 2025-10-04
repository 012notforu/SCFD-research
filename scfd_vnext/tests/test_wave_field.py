import numpy as np

from benchmarks.wave_field import (
    WaveFieldParams,
    WaveFieldControlConfig,
    WaveFieldController,
    WaveFieldSimulator,
    synthetic_wave_target,
)


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
