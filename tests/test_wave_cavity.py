import numpy as np

from benchmarks.wave_field_cavity import (
    WaveCavityParams,
    WaveCavityControlConfig,
    WaveCavityController,
    WaveCavitySimulator,
    standing_mode_target,
)


def test_standing_mode_target_shape():
    target = standing_mode_target((32, 32), mode_m=2, mode_n=3)
    assert target.shape == (32, 32)
    assert np.max(np.abs(target)) <= 1.0


def test_wave_cavity_step_metrics():
    params = WaveCavityParams(shape=(48, 48), mode_m=1, mode_n=2, dt=0.04)
    cfg = WaveCavityControlConfig(control_gain=0.015, control_clip=0.08)
    controller = WaveCavityController(cfg, params.shape)
    target = standing_mode_target(params.shape, params.mode_m, params.mode_n)
    sim = WaveCavitySimulator(params, controller, target)
    stats = sim.step()
    expected = {"mse", "energy", "boundary_energy"}
    assert expected.issubset(stats.keys())
    assert np.isfinite(stats["mse"])

