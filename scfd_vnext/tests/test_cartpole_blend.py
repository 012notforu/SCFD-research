import numpy as np
import pytest

from benchmarks.scfd_cartpole import SCFDControllerConfig, SCFDCartPoleController


def make_controller() -> SCFDCartPoleController:
    cfg = SCFDControllerConfig()
    controller = SCFDCartPoleController(cfg, rng=np.random.default_rng(0))
    controller.reset()
    return controller


def test_blend_respects_linear_weight_only():
    controller = make_controller()
    controller.cfg.blend_linear_weight = 1.0
    controller.cfg.blend_ternary_weight = 0.0
    result = controller._blend_actions(0.75, -3.0)
    assert result == pytest.approx(0.75)
    assert controller.last_blend_components["linear"] == pytest.approx(0.75)
    assert controller.last_blend_components["ternary"] == pytest.approx(-3.0)
    assert controller.last_action == pytest.approx(result)


def test_blend_clips_combined_action():
    controller = make_controller()
    controller.cfg.blend_linear_weight = 0.5
    controller.cfg.blend_ternary_weight = 1.0
    controller.cfg.action_clip = 1.5
    result = controller._blend_actions(1.0, 2.0)
    assert result == pytest.approx(1.5)
    assert controller.last_action == pytest.approx(1.5)
