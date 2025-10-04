import numpy as np

from run.train_cma_wave import _vector_from_config, _config_from_vector
from benchmarks.wave_field import WaveFieldControlConfig, WaveFieldParams


def test_wave_vector_roundtrip():
    control = WaveFieldControlConfig(
        encode_gain=0.7,
        encode_decay=0.85,
        control_gain=0.01,
        control_clip=0.05,
        smooth_lambda=0.4,
        theta_clip=1.8,
    )
    params = WaveFieldParams(wave_speed=1.5)
    vec = _vector_from_config(control, params)
    control2, params2 = _config_from_vector(control, params, vec)
    vec2 = _vector_from_config(control2, params2)
    np.testing.assert_allclose(vec, vec2, rtol=1e-6, atol=1e-6)
