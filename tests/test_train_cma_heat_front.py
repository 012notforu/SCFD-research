import json
from pathlib import Path

import numpy as np

from benchmarks.heat_diffusion import HeatDiffusionControlConfig
from benchmarks.heat_front_tracking import HeatFrontParams
from run.train_cma_heat_front import (
    _config_from_vector,
    _metadata_from_params,
    _save_vector,
    _vector_from_config,
)


def test_heat_front_vector_roundtrip() -> None:
    control = HeatDiffusionControlConfig(
        encode_gain=0.55,
        encode_decay=0.88,
        control_gain=0.0025,
        control_clip=0.08,
        smooth_lambda=0.45,
        theta_clip=2.2,
    )
    params = HeatFrontParams(alpha=0.18, front_radius=0.3)
    vec = _vector_from_config(control, params)
    control2, params2 = _config_from_vector(control, params, vec)
    vec2 = _vector_from_config(control2, params2)
    np.testing.assert_allclose(vec, vec2, rtol=1e-6, atol=1e-6)


def test_heat_front_metadata_serialization(tmp_path: Path) -> None:
    params = HeatFrontParams(
        shape=(40, 40),
        alpha=0.2,
        dt=0.07,
        noise=0.02,
        front_radius=0.28,
        front_width=0.015,
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
    assert data["front_radius"] == params.front_radius
    assert data["front_width"] == params.front_width
