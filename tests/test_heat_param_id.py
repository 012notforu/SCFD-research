import numpy as np

from benchmarks.heat_parameter_id import HeatParameterIDParams, HeatParameterIDSimulator
from benchmarks.heat_diffusion import HeatDiffusionControlConfig, HeatDiffusionController


def test_heat_param_id_step_metrics() -> None:
    params = HeatParameterIDParams(shape=(24, 24), split_axis="horizontal")
    controller = HeatDiffusionController(HeatDiffusionControlConfig(control_gain=0.002), params.shape)
    sim = HeatParameterIDSimulator(params, controller)
    stats = sim.step()
    assert {"mse", "energy", "alpha_rmse"}.issubset(stats)
    assert np.isfinite(stats["alpha_rmse"])


def test_heat_param_id_run_history() -> None:
    params = HeatParameterIDParams(shape=(16, 16))
    controller = HeatDiffusionController(HeatDiffusionControlConfig(), params.shape)
    sim = HeatParameterIDSimulator(params, controller)
    history = sim.run(steps=12, record_interval=4)
    assert history["alpha_history"].shape[0] >= 3
