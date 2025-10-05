import numpy as np

from benchmarks.heat_diffusion import HeatDiffusionControlConfig
from benchmarks.heat_diffusion_inverse import HeatInverseParams
from run.train_cma_heat_inverse import _config_from_vector, _vector_from_config


def test_heat_inverse_vector_roundtrip():
    control = HeatDiffusionControlConfig(
        encode_gain=0.6,
        encode_decay=0.88,
        control_gain=0.0025,
        control_clip=0.06,
        smooth_lambda=0.35,
        theta_clip=1.9,
    )
    params = HeatInverseParams(alpha=0.22, control_budget=4.0)
    vec = _vector_from_config(control, params)
    control2, params2 = _config_from_vector(control, params, vec)
    vec2 = _vector_from_config(control2, params2)
    np.testing.assert_allclose(vec, vec2, rtol=1e-6, atol=1e-6)
