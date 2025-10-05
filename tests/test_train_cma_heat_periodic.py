import numpy as np

from benchmarks.heat_diffusion import HeatDiffusionControlConfig
from benchmarks.heat_diffusion_periodic import HeatPeriodicParams
from run.train_cma_heat_periodic import _config_from_vector, _vector_from_config


def test_heat_periodic_vector_roundtrip():
    control = HeatDiffusionControlConfig(
        encode_gain=0.9,
        encode_decay=0.8,
        control_gain=0.003,
        control_clip=0.04,
        smooth_lambda=0.45,
        theta_clip=2.6,
    )
    params = HeatPeriodicParams(alpha=0.25, dt=0.09, dt_jitter=0.01, control_budget=8.0)
    vec = _vector_from_config(control, params)
    control2, params2 = _config_from_vector(control, params, vec)
    vec2 = _vector_from_config(control2, params2)
    np.testing.assert_allclose(vec, vec2, rtol=1e-6, atol=1e-6)
