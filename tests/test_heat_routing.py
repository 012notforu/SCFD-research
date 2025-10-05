import numpy as np
import pytest

from benchmarks.heat_diffusion import HeatDiffusionControlConfig, HeatDiffusionController
from benchmarks.heat_diffusion_routing import (
    HeatDiffusionRoutingParams,
    HeatDiffusionRoutingSimulator,
    generate_blob_pattern,
)


def test_generate_blob_pattern_shapes() -> None:
    pattern = generate_blob_pattern((32, 32), [(0.25, 0.25), (0.75, 0.75)], sigma=0.03)
    assert pattern.shape == (32, 32)
    assert np.max(pattern) <= 1.0


def test_heat_routing_step_metrics() -> None:
    params = HeatDiffusionRoutingParams(shape=(24, 24), initial_centers=((0.3, 0.3), (0.7, 0.7)), target_centers=((0.7, 0.3), (0.3, 0.7)))
    controller = HeatDiffusionController(HeatDiffusionControlConfig(control_gain=0.002), params.shape)
    sim = HeatDiffusionRoutingSimulator(params, controller)
    stats = sim.step()
    assert {"mse", "energy", "collision_penalty"}.issubset(stats)
    assert np.isfinite(stats["collision_penalty"])


def test_heat_routing_run_history() -> None:
    params = HeatDiffusionRoutingParams(shape=(16, 16))
    controller = HeatDiffusionController(HeatDiffusionControlConfig(), params.shape)
    sim = HeatDiffusionRoutingSimulator(params, controller)
    history = sim.run(steps=10, record_interval=3)
    assert "history" in history
    assert history["history"].shape[0] >= 3
