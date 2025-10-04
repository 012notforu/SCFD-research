import numpy as np

from benchmarks.heat_diffusion import (
    HeatDiffusionParams,
    HeatDiffusionControlConfig,
    HeatDiffusionController,
    HeatDiffusionSimulator,
    synthetic_temperature,
)


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
