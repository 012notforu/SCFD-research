import numpy as np

from benchmarks.flow_cylinder import FlowCylinderControlConfig, FlowCylinderController
from benchmarks.flow_constriction import FlowConstrictionParams, FlowConstrictionSimulator


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

