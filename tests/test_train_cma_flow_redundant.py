import numpy as np

from benchmarks.flow_cylinder import FlowCylinderControlConfig
from benchmarks.flow_redundant import FlowRedundantParams
from run.train_cma_flow_redundant import _config_from_vector, _vector_from_config


def test_flow_redundant_vector_roundtrip():
    control = FlowCylinderControlConfig(
        encode_gain=0.55,
        encode_decay=0.82,
        control_gain=0.012,
        control_clip=0.09,
        smooth_lambda=0.4,
        theta_clip=1.7,
    )
    params = FlowRedundantParams(viscosity=0.028, control_budget=2.5)
    vec = _vector_from_config(control, params)
    control2, params2 = _config_from_vector(control, params, vec)
    vec2 = _vector_from_config(control2, params2)
    np.testing.assert_allclose(vec, vec2, rtol=1e-6, atol=1e-6)
