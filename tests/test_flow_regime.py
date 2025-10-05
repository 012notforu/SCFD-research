import numpy as np

from benchmarks.flow_cylinder import FlowCylinderControlConfig
from benchmarks.flow_regime_sweep import FlowRegimeParams, FlowRegimeSweep


def test_flow_regime_sweep_metrics():
    params = FlowRegimeParams(shape=(48, 64), inflow_values=(0.7, 1.0), steps_per_regime=100, record_interval=20)
    cfg = FlowCylinderControlConfig(control_gain=0.02, control_clip=0.08)
    sweep = FlowRegimeSweep(params, cfg)
    result = sweep.run(seeds=[0])
    assert len(result["metrics"]) == 2
    metrics = sweep.aggregate_metrics()
    assert "mean_wake_mse" in metrics
    assert np.isfinite(metrics["mean_wake_mse"])
