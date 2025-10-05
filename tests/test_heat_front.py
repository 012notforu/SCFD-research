import numpy as np

from benchmarks.heat_front_tracking import (
    HeatFrontParams,
    HeatFrontTrackingSimulator,
    generate_ring,
)
from benchmarks.heat_diffusion import HeatDiffusionControlConfig, HeatDiffusionController


def test_generate_ring_stats() -> None:
    ring = generate_ring((32, 32), radius=0.25, width=0.02)
    assert ring.shape == (32, 32)
    assert np.max(ring) <= 1.0


def test_heat_front_step_metrics() -> None:
    params = HeatFrontParams(shape=(24, 24), front_radius=0.2)
    controller = HeatDiffusionController(HeatDiffusionControlConfig(control_gain=0.002), params.shape)
    sim = HeatFrontTrackingSimulator(params, controller)
    stats = sim.step()
    assert {"mse", "energy", "curvature_proxy"}.issubset(stats)
    assert np.isfinite(stats["curvature_proxy"])


def test_heat_front_run_history() -> None:
    params = HeatFrontParams(shape=(16, 16))
    controller = HeatDiffusionController(HeatDiffusionControlConfig(), params.shape)
    sim = HeatFrontTrackingSimulator(params, controller)
    history = sim.run(steps=15, record_interval=5)
    assert history["history"].shape[0] >= 3
