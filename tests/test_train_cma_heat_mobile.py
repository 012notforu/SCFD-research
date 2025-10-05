import numpy as np

from benchmarks.heat_diffusion import HeatDiffusionControlConfig
from benchmarks.heat_diffusion_mobile import HeatMobileParams
from run.train_cma_heat_mobile import _config_from_vector, _vector_from_config


def test_heat_mobile_vector_roundtrip():
    control = HeatDiffusionControlConfig(
        encode_gain=0.7,
        encode_decay=0.85,
        control_gain=0.003,
        control_clip=0.08,
        smooth_lambda=0.45,
        theta_clip=2.4,
    )
    params = HeatMobileParams(heater_amplitude=0.8, heater_radius=5, alpha=0.21)
    vec = _vector_from_config(control, params)
    control2, params2 = _config_from_vector(control, params, vec)
    vec2 = _vector_from_config(control2, params2)
    np.testing.assert_allclose(vec, vec2, rtol=1e-6, atol=1e-6)
