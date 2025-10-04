import numpy as np

from benchmarks.heat_diffusion import HeatDiffusionControlConfig, HeatDiffusionController
from benchmarks.heat_diffusion_anisotropic import (
    HeatAnisotropicParams,
    HeatAnisotropicSimulator,
    synthetic_anisotropic_temperature,
)


def test_synthetic_anisotropic_temperature_shapes():
    pattern = synthetic_anisotropic_temperature((32, 32), kind="elliptic_hotspot", angle=0.4)
    assert pattern.shape == (32, 32)
    assert np.max(pattern) <= 1.0


def test_heat_anisotropic_step_metrics():
    params = HeatAnisotropicParams(shape=(24, 24), alpha_major=0.28, alpha_minor=0.12, orientation=0.3)
    cfg = HeatDiffusionControlConfig(encode_gain=0.25, control_gain=0.002)
    controller = HeatDiffusionController(cfg, params.shape)
    target = synthetic_anisotropic_temperature(params.shape, kind="tilted", angle=0.2)
    sim = HeatAnisotropicSimulator(params, controller, target)
    stats = sim.step()
    expected = {"mse", "energy", "delta_energy", "control_norm", "principal_ratio", "orientation"}
    assert expected.issubset(stats.keys())
    assert stats["principal_ratio"] >= 1.0

