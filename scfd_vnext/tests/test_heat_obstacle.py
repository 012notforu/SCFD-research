import numpy as np

from benchmarks.heat_diffusion import HeatDiffusionControlConfig, HeatDiffusionController
from benchmarks.heat_diffusion_obstacle import (
    HeatObstacleParams,
    HeatObstacleSimulator,
    synthetic_obstacle_target,
)


def test_heat_obstacle_mask_shapes():
    params = HeatObstacleParams(shape=(48, 48), gap_height=8, gap_width=3)
    controller = HeatDiffusionController(HeatDiffusionControlConfig(), params.shape)
    target = synthetic_obstacle_target(params.shape, kind="hot_corner")
    sim = HeatObstacleSimulator(params, controller, target)
    assert sim.obstacle_mask.shape == params.shape
    assert sim.obstacle_mask.any()


def test_heat_obstacle_step_budget():
    params = HeatObstacleParams(shape=(24, 24), control_budget=0.5)
    cfg = HeatDiffusionControlConfig(control_gain=0.01, control_clip=0.2)
    controller = HeatDiffusionController(cfg, params.shape)
    target = synthetic_obstacle_target(params.shape, kind="hot_corner")
    sim = HeatObstacleSimulator(params, controller, target)
    stats = sim.step()
    assert stats["control_total_l1"] <= params.control_budget + 1e-5
    assert np.isfinite(stats["corner_mse"])

