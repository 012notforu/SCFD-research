import numpy as np

from benchmarks.wave_field_partial import (
    WavePartialParams,
    WavePartialControlConfig,
    WavePartialController,
    WavePartialSimulator,
    random_sensor_mask,
)
from benchmarks.wave_field import synthetic_wave_target


def test_random_sensor_mask_fraction():
    rng = np.random.default_rng(0)
    mask = random_sensor_mask((20, 20), 0.25, rng)
    coverage = np.mean(mask)
    assert 0.1 < coverage < 0.4


def test_wave_partial_step_metrics():
    params = WavePartialParams(shape=(32, 32), sensor_fraction=0.3, action_delay=2)
    cfg = WavePartialControlConfig(control_gain=0.015, control_clip=0.08)
    controller = WavePartialController(cfg, params.shape, params.action_delay)
    rng = np.random.default_rng(1)
    mask = random_sensor_mask(params.shape, params.sensor_fraction, rng)
    target = synthetic_wave_target(params.shape, kind="focus")
    sim = WavePartialSimulator(params, controller, target, mask)
    stats = sim.step()
    expected = {"mse", "sensed_mse", "energy", "control_norm", "sensor_coverage"}
    assert expected.issubset(stats.keys())
    assert np.isfinite(stats["mse"])
    assert stats["sensor_coverage"] > 0.0

