import numpy as np

from benchmarks.scfd_cartpole import SCFDControllerConfig
from run.train_cma_scfd import _config_from_vector, _evaluate, _vector_from_config


def test_vector_roundtrip():
    base = SCFDControllerConfig()
    vec = _vector_from_config(base)
    rebuilt = _vector_from_config(_config_from_vector(base, vec))
    np.testing.assert_allclose(vec, rebuilt)


def test_config_clipping():
    base = SCFDControllerConfig()
    vec = _vector_from_config(base)
    vec += 100.0  # push out of range
    cfg = _config_from_vector(base, vec)
    assert cfg.blend_linear_weight <= 2.0
    assert cfg.blend_ternary_weight <= 2.0
    assert 2.0 <= cfg.ternary_force_scale <= 15.0
    assert 0.05 <= cfg.ternary_smooth_lambda <= 1.0


def test_evaluate_runs_quick():
    base = SCFDControllerConfig()
    seeds = [0, 1]
    score, metrics = _evaluate(base, seeds, steps=200)
    assert np.isfinite(score)
    assert metrics["mean_steps"] >= 0
