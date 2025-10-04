import numpy as np

from benchmarks.flow_cylinder import (
    FlowCylinderParams,
    FlowCylinderControlConfig,
    FlowCylinderController,
    FlowCylinderSimulator,
)


def test_flow_masks_and_reset():
    params = FlowCylinderParams(shape=(48, 48), cylinder_radius=6)
    controller = FlowCylinderController(FlowCylinderControlConfig(), params.shape)
    sim = FlowCylinderSimulator(params, controller)
    assert sim.cylinder_mask.shape == params.shape
    assert sim.wake_mask.any()
    assert sim.u.shape == params.shape


def test_flow_step_metrics_finite():
    params = FlowCylinderParams(shape=(48, 48), cylinder_radius=6)
    controller = FlowCylinderController(FlowCylinderControlConfig(control_gain=0.01), params.shape)
    sim = FlowCylinderSimulator(params, controller)
    metrics = sim.step()
    for value in metrics.values():
        assert np.isfinite(value)

