import json
from pathlib import Path

import numpy as np

from benchmarks.heat_diffusion import HeatDiffusionControlConfig
from benchmarks.heat_diffusion_routing import HeatDiffusionRoutingParams
from run.train_cma_heat_routing import (
    _config_from_vector,
    _metadata_from_params,
    _save_vector,
    _vector_from_config,
)


def test_heat_routing_vector_roundtrip() -> None:
    control = HeatDiffusionControlConfig(
        encode_gain=0.5,
        encode_decay=0.9,
        control_gain=0.002,
        control_clip=0.07,
        smooth_lambda=0.4,
        theta_clip=2.1,
    )
    params = HeatDiffusionRoutingParams(alpha=0.2, collision_radius=0.1)
    vec = _vector_from_config(control, params)
    control2, params2 = _config_from_vector(control, params, vec)
    vec2 = _vector_from_config(control2, params2)
    np.testing.assert_allclose(vec, vec2, rtol=1e-6, atol=1e-6)


def test_heat_routing_metadata_serialization(tmp_path: Path) -> None:
    params = HeatDiffusionRoutingParams(
        shape=(48, 48),
        alpha=0.19,
        dt=0.08,
        noise=0.015,
        initial_centers=((0.25, 0.25), (0.75, 0.75)),
        target_centers=((0.75, 0.25), (0.25, 0.75)),
        blob_sigma=0.03,
        collision_radius=0.09,
    )
    metadata = _metadata_from_params(params)
    vector = np.zeros(8, dtype=np.float32)
    metrics = {"mean_mse": 0.1}
    out_path = tmp_path / "vector.json"
    _save_vector(out_path, vector, metrics, metadata=metadata)

    data = json.loads(out_path.read_text())
    assert data["shape"] == list(params.shape)
    assert data["alpha"] == params.alpha
    assert data["dt"] == params.dt
    assert data["noise"] == params.noise
    assert data["initial_centers"] == [list(center) for center in params.initial_centers]
    assert data["target_centers"] == [list(center) for center in params.target_centers]
    assert data["blob_sigma"] == params.blob_sigma
    assert data["collision_radius"] == params.collision_radius

    data = json.loads(out_path.read_text())
    assert data["initial_centers"] == [list(center) for center in params.initial_centers]
    assert data["target_centers"] == [list(center) for center in params.target_centers]
    assert data["blob_sigma"] == params.blob_sigma
    assert data["collision_radius"] == params.collision_radius
