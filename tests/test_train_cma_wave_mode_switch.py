import numpy as np

from benchmarks.wave_field import WaveFieldControlConfig
from benchmarks.wave_field_mode_switch import WaveModeSwitchParams
from run.train_cma_wave_mode_switch import _config_from_vector, _vector_from_config


def test_wave_mode_switch_vector_roundtrip():
    control = WaveFieldControlConfig(
        encode_gain=0.7,
        encode_decay=0.87,
        control_gain=0.018,
        control_clip=0.12,
        smooth_lambda=0.42,
        theta_clip=2.1,
    )
    params = WaveModeSwitchParams(wave_speed=1.3, switch_step=400, initial_kind="focus", switch_kind="defocus")
    vec = _vector_from_config(control, params)
    control2, params2 = _config_from_vector(control, params, vec)
    vec2 = _vector_from_config(control2, params2)
    np.testing.assert_allclose(vec, vec2, rtol=1e-6, atol=1e-6)
