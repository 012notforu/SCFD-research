import numpy as np

from run.train_cma_gray_scott import _config_from_vector, _vector_from_config
from benchmarks.gray_scott import GrayScottControlConfig, GrayScottParams


def test_vector_roundtrip_gray_scott():
    control = GrayScottControlConfig(
        encode_gain=0.7,
        encode_decay=0.8,
        control_gain_feed=0.001,
        control_gain_kill=0.0015,
        control_clip=0.02,
        smooth_lambda=0.4,
        theta_clip=2.5,
    )
    params = GrayScottParams(F=0.04, k=0.06)
    vec = _vector_from_config(control, params)
    control2, params2 = _config_from_vector(control, params, vec)
    vec2 = _vector_from_config(control2, params2)
    np.testing.assert_allclose(vec, vec2, rtol=1e-6, atol=1e-6)
