import numpy as np

from benchmarks.heat_diffusion import HeatDiffusionControlConfig, HeatDiffusionController
from benchmarks.heat_diffusion_periodic import (
    HeatPeriodicParams,
    HeatPeriodicSimulator,
    synthetic_periodic_temperature,
)


def test_synthetic_periodic_wraps_cleanly():
    pattern = synthetic_periodic_temperature((32, 32), kind="checker")
    assert pattern.shape == (32, 32)
    np.testing.assert_allclose(pattern[0, :], pattern[-1, :], atol=1e-6)
    np.testing.assert_allclose(pattern[:, 0], pattern[:, -1], atol=1e-6)


def test_heat_periodic_step_metrics():
    params = HeatPeriodicParams(shape=(24, 24), init_seed=2, dt=0.08, control_budget=2.5)
    cfg = HeatDiffusionControlConfig(encode_gain=0.25, control_gain=0.0015, control_clip=0.05)
    controller = HeatDiffusionController(cfg, params.shape)
    target = synthetic_periodic_temperature(params.shape, kind="stripe")
    sim = HeatPeriodicSimulator(params, controller, target)
    stats = sim.step()
    expected_keys = {
        "mse",
        "energy",
        "delta_energy",
        "boundary_wrap_mse",
        "control_norm",
        "control_total_l1",
        "control_clip_fraction",
        "budget_utilisation",
        "controller_latency_ms",
    }
    assert expected_keys.issubset(stats.keys())
    assert stats["control_total_l1"] <= params.control_budget + 1e-5


def test_heat_periodic_budget_scaling():
    params = HeatPeriodicParams(shape=(16, 16), init_seed=3, control_budget=0.2)
    cfg = HeatDiffusionControlConfig(control_gain=0.01, control_clip=0.5)
    controller = HeatDiffusionController(cfg, params.shape)
    target = synthetic_periodic_temperature(params.shape, kind="mixed")
    sim = HeatPeriodicSimulator(params, controller, target)
    stats = sim.step()
    assert stats["control_total_l1"] <= 0.2 + 1e-5
