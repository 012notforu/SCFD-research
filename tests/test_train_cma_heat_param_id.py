import json
from pathlib import Path

import numpy as np

from benchmarks.heat_diffusion import HeatDiffusionControlConfig
from benchmarks.heat_parameter_id import HeatParameterIDParams
from run.train_cma_heat_param_id import (
    _config_from_vector,
    _metadata_from_params,
    _save_vector,
    _vector_from_config,
)


def test_heat_param_id_vector_roundtrip() -> None:
    control = HeatDiffusionControlConfig(
        encode_gain=0.6,
        encode_decay=0.9,
        control_gain=0.002,
        control_clip=0.07,
        smooth_lambda=0.4,
        theta_clip=2.0,
    )
    params = HeatParameterIDParams(alpha=0.2, split_axis="vertical")
    vec = _vector_from_config(control, params)
    control2, params2 = _config_from_vector(control, params, vec)
    vec2 = _vector_from_config(control2, params2)
    np.testing.assert_allclose(vec, vec2, rtol=1e-6, atol=1e-6)


def test_heat_param_id_metadata_serialization(tmp_path: Path) -> None:
    params = HeatParameterIDParams(
        shape=(32, 64),
        alpha=0.21,
        dt=0.06,
        noise=0.012,
        alpha_low=0.11,
        alpha_high=0.29,
        split_axis="horizontal",
    )
    metadata = _metadata_from_params(params)
    vector = np.zeros(7, dtype=np.float32)
    metrics = {"mean_mse": 0.1}
    out_path = tmp_path / "vector.json"
    _save_vector(out_path, vector, metrics, metadata=metadata)

    data = json.loads(out_path.read_text())
    assert data["shape"] == list(params.shape)
    assert data["alpha"] == params.alpha
    assert data["dt"] == params.dt
    assert data["noise"] == params.noise
    assert data["alpha_low"] == params.alpha_low
    assert data["alpha_high"] == params.alpha_high
    assert data["split_axis"] == params.split_axis
