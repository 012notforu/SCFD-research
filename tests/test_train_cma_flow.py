import numpy as np

from run.train_cma_flow import _vector_from_config, _config_from_vector
from benchmarks.flow_cylinder import FlowCylinderControlConfig, FlowCylinderParams


def test_flow_vector_roundtrip():
    control = FlowCylinderControlConfig(
        encode_gain=0.6,
        encode_decay=0.8,
        control_gain=0.01,
        control_clip=0.05,
        smooth_lambda=0.4,
        theta_clip=1.8,
    )
    params = FlowCylinderParams(viscosity=0.03)
    vec = _vector_from_config(control, params)
    control2, params2 = _config_from_vector(control, params, vec)
    vec2 = _vector_from_config(control2, params2)
    np.testing.assert_allclose(vec, vec2, rtol=1e-6, atol=1e-6)
