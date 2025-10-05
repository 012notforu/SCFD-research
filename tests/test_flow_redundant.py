import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.flow_cylinder import FlowCylinderControlConfig, FlowCylinderController
from benchmarks.flow_redundant import FlowRedundantParams, FlowRedundantSimulator
from run.train_cma_flow_redundant import _config_from_vector


def test_flow_redundant_actuator_masks_normalized():
    params = FlowRedundantParams(shape=(48, 48), actuator_rows=(24,), actuator_cols=(16, 32))
    controller = FlowCylinderController(FlowCylinderControlConfig(), params.shape)
    sim = FlowRedundantSimulator(params, controller)
    assert len(sim.actuator_masks) == len(params.actuator_rows) * len(params.actuator_cols)
    totals = [mask.sum() for mask in sim.actuator_masks]
    assert all(np.isclose(total, 1.0) for total in totals)


def test_flow_redundant_step_metrics():
    params = FlowRedundantParams(shape=(48, 64), actuator_rows=(24,), actuator_cols=(20, 40), control_budget=1.0)
    cfg = FlowCylinderControlConfig(control_gain=0.01, control_clip=0.05)
    controller = FlowCylinderController(cfg, params.shape)
    sim = FlowRedundantSimulator(params, controller)
    stats = sim.step()
    assert {"throughput", "energy", "budget_util", "actuator_rms"}.issubset(stats)
    assert 0.0 <= stats["budget_util"] <= 1.0



@pytest.mark.slow
def test_flow_redundant_best_vector_regression():
    vector_path = Path('runs/flow_redundant_cma/best_vector.json')
    data = json.loads(vector_path.read_text())
    vector = np.asarray(data['vector'], dtype=np.float32)

    base_control = FlowCylinderControlConfig()
    base_params = FlowRedundantParams()
    control_cfg, params = _config_from_vector(base_control, base_params, vector)

    controller = FlowCylinderController(control_cfg, params.shape)
    simulator = FlowRedundantSimulator(params, controller)
    history = simulator.run(steps=1400, record_interval=50)
    final_metrics = history['metrics'][-1]

    assert final_metrics['throughput'] >= data['metrics']['mean_throughput'] * 0.97
    assert np.isfinite(final_metrics['energy'])
    assert final_metrics['energy'] <= data['metrics']['mean_energy'] * 1.5
    assert final_metrics['budget_util'] <= max(1.0, data['metrics']['mean_budget'] + 0.1)
    assert final_metrics['actuator_rms'] <= data['metrics']['mean_actuator_rms'] + 1e-4
