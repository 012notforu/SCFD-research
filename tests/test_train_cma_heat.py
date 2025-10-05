import numpy as np

from run.train_cma_heat import _vector_from_config, _config_from_vector
from benchmarks.heat_diffusion import HeatDiffusionControlConfig, HeatDiffusionParams


def test_heat_vector_roundtrip():
    control = HeatDiffusionControlConfig(
        encode_gain=0.8,
        encode_decay=0.7,
        control_gain=0.002,
        control_clip=0.03,
        smooth_lambda=0.5,
        theta_clip=2.2,
    )
    params = HeatDiffusionParams(alpha=0.22)
    vec = _vector_from_config(control, params)
    control2, params2 = _config_from_vector(control, params, vec)
    vec2 = _vector_from_config(control2, params2)
    np.testing.assert_allclose(vec, vec2, rtol=1e-6, atol=1e-6)
