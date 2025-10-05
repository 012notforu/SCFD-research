import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.heat_diffusion import HeatDiffusionControlConfig
from benchmarks.heat_diffusion_arc import HeatDiffusionArcParams
from run.train_cma_heat_arc import (
    _config_from_vector,
    _metadata_from_params,
    _save_vector,
    _vector_from_config,
)


def test_heat_arc_vector_roundtrip() -> None:
    control = HeatDiffusionControlConfig(
        encode_gain=0.6,
        encode_decay=0.88,
        control_gain=0.0015,
        control_clip=0.06,
        smooth_lambda=0.4,
        theta_clip=1.9,
    )
    params = HeatDiffusionArcParams(
        alpha=0.22,
        transform_cycle=("identity", "rotate90", "flip_horizontal"),
        transform_cycle_interval=120,
    )
    vec = _vector_from_config(control, params)
    control2, params2 = _config_from_vector(control, params, vec)
    vec2 = _vector_from_config(control2, params2)
    np.testing.assert_allclose(vec, vec2, rtol=1e-6, atol=1e-6)

def test_heat_arc_metadata_serialization(tmp_path: Path) -> None:
    params = HeatDiffusionArcParams(
        transform_cycle=("identity", "rotate90"),
        transform_cycle_interval=50,
        base_target_kind="gradient",
    )
    metadata = _metadata_from_params(params)
    vector = np.zeros(8, dtype=np.float32)
    metrics = {"mean_mse": 0.1}
    out_path = tmp_path / "vector.json"
    _save_vector(out_path, vector, metrics, metadata=metadata)

    data = json.loads(out_path.read_text())
    assert data["transform_cycle"] == list(params.transform_cycle)
    assert data["transform_cycle_interval"] == params.transform_cycle_interval
    assert data["base_target_kind"] == params.base_target_kind
    assert data["alpha"] == pytest.approx(params.alpha)
