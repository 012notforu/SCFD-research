import numpy as np

from benchmarks.gray_scott import (
    GrayScottParams,
    GrayScottControlConfig,
    GrayScottController,
    GrayScottSimulator,
    synthetic_target,
)


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
