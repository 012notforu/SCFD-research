import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.flow_cylinder import FlowCylinderControlConfig, FlowCylinderController
from benchmarks.flow_constriction import FlowConstrictionParams, FlowConstrictionSimulator
from run.train_cma_flow_constriction import _config_from_vector


def test_flow_constriction_geometry():
    params = FlowConstrictionParams(shape=(64, 64), slit_height=16, constriction_half_width=5)
    controller = FlowCylinderController(FlowCylinderControlConfig(), params.shape)
    sim = FlowConstrictionSimulator(params, controller)
    assert sim.solid_mask.shape == params.shape
    assert sim.actuator_mask.shape == params.shape
    assert sim.monitor_mask.any()


def test_flow_constriction_step_metrics():
    params = FlowConstrictionParams(shape=(48, 64), slit_height=14, constriction_half_width=4)
    cfg = FlowCylinderControlConfig(control_gain=0.02, control_clip=0.08)
    controller = FlowCylinderController(cfg, params.shape)
    sim = FlowConstrictionSimulator(params, controller)
    stats = sim.step()
    expected = {"throughput", "backflow", "energy", "delta_energy", "control_norm"}
    assert expected.issubset(stats.keys())
    assert np.isfinite(stats["throughput"])


@pytest.mark.slow
def test_flow_constriction_best_vector_regression():
    vector_path = Path("runs/flow_constriction_cma/best_vector.json")
    data = json.loads(vector_path.read_text())
    vector = np.asarray(data["vector"], dtype=np.float32)

    base_control = FlowCylinderControlConfig()
    base_params = FlowConstrictionParams()
    control_cfg, params = _config_from_vector(base_control, base_params, vector)

    controller = FlowCylinderController(control_cfg, params.shape)
    simulator = FlowConstrictionSimulator(params, controller)
    history = simulator.run(steps=1600, record_interval=60)
    final_metrics = history["metrics"][-1]

    assert final_metrics["throughput"] >= data["metrics"]["mean_throughput"] * 0.97
    assert final_metrics["backflow"] <= max(0.02, data["metrics"]["mean_backflow"] + 0.01)
    assert np.isfinite(final_metrics["energy"])
    assert final_metrics["energy"] <= data["metrics"]["mean_energy"] * 1.2
