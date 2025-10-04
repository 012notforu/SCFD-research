import numpy as np
import pytest

from engine.params import load_config
from observe.features import CartPoleFeatureExtractor
from logic_ternary import (
    CartPoleTernaryConfig,
    CartPoleTernaryController,
    hysteretic_sign_trit,
    ternary_majority,
)


def test_hysteretic_sign_trit_sticky():
    hi, lo = 1e-3, 5e-4
    first = hysteretic_sign_trit(np.array([2e-3], dtype=np.float32), hi=hi, lo=lo)
    assert first[0] == 1
    second = hysteretic_sign_trit(np.array([2e-4], dtype=np.float32), hi=hi, lo=lo, previous=first)
    assert second[0] == 0
    third = hysteretic_sign_trit(np.array([-2e-3], dtype=np.float32), hi=hi, lo=lo, previous=second)
    assert third[0] == -1


def test_ternary_majority_prefers_nonzero():
    a = np.array([1, -1, 0], dtype=np.int8)
    b = np.array([1, 0, 0], dtype=np.int8)
    c = np.array([0, 0, 0], dtype=np.int8)
    result = ternary_majority(a, b, c)
    np.testing.assert_array_equal(result, np.array([1, -1, 0], dtype=np.int8))
    single = ternary_majority(1, -1, 0)
    assert int(np.asarray(single, dtype=np.int8)) == 0


def test_cartpole_ternary_controller_direction():
    cfg = load_config("cfg/defaults.yaml")
    extractor = CartPoleFeatureExtractor(cfg, momentum=0.2)
    controller = CartPoleTernaryController(
        extractor,
        CartPoleTernaryConfig(force_scale=1.0, smooth_lambda=1.0, action_clip=5.0),
    )
    controller.reset()
    theta = np.zeros(cfg.grid.shape, dtype=np.float32)
    theta_dot = np.zeros_like(theta)
    theta[:, cfg.grid.shape[1] // 2 :] = 0.25
    env = np.array([0.0, 0.0, 0.2, 0.0], dtype=np.float32)
    action, info = controller.compute_action(theta, theta_dot, env)
    assert info["direction_trit"] == -1
    assert action < 0

    controller.reset()
    theta[:, :] = 0.0
    theta[:, : cfg.grid.shape[1] // 2] = -0.25
    env = np.array([0.0, 0.0, -0.2, 0.0], dtype=np.float32)
    action, info = controller.compute_action(theta, theta_dot, env)
    assert info["direction_trit"] == 1
    assert action > 0
    assert abs(action) <= controller.config.action_clip

def test_cartpole_ternary_controller_accepts_feature_vector():
    cfg = load_config("cfg/defaults.yaml")
    base_cfg = CartPoleTernaryConfig(force_scale=1.0, smooth_lambda=1.0, action_clip=5.0)
    extractor = CartPoleFeatureExtractor(cfg, momentum=0.2)
    controller = CartPoleTernaryController(extractor, base_cfg)
    controller.reset()
    theta = np.zeros(cfg.grid.shape, dtype=np.float32)
    theta_dot = np.zeros_like(theta)
    theta[:, cfg.grid.shape[1] // 2 :] = 0.2
    env = np.array([0.0, 0.0, 0.15, 0.0], dtype=np.float32)
    feature_vector = extractor.extract(theta, theta_dot, env, prev_action=controller.prev_action)

    action_cached, info_cached = controller.compute_action(
        theta,
        theta_dot,
        env,
        feature_vector=feature_vector,
    )

    secondary_extractor = CartPoleFeatureExtractor(cfg, momentum=0.2)
    controller_direct = CartPoleTernaryController(secondary_extractor, base_cfg)
    controller_direct.reset()
    action_direct, info_direct = controller_direct.compute_action(theta, theta_dot, env)

    assert action_cached == pytest.approx(action_direct)
    assert info_cached["direction_trit"] == info_direct["direction_trit"]
