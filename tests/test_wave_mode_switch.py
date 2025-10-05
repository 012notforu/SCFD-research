import json
from pathlib import Path

import numpy as np
import pytest
from collections import Counter

from benchmarks.wave_field import WaveFieldControlConfig, WaveFieldController
from benchmarks.wave_field_mode_switch import WaveModeSwitchParams, WaveModeSwitchSimulator
from run.train_cma_wave_mode_switch import _config_from_vector


def test_wave_mode_switch_phases():
    params = WaveModeSwitchParams(shape=(32, 32), switch_step=3, dt=0.05)
    controller = WaveFieldController(WaveFieldControlConfig(control_gain=0.01), params.shape)
    sim = WaveModeSwitchSimulator(params, controller)
    phases = []
    for _ in range(8):
        stats = sim.step()
        phases.append(stats["phase"])
    counts = Counter(phases)
    assert counts["initial"] > 0
    assert counts["switched"] > 0


@pytest.mark.slow
def test_wave_mode_switch_best_vector_regression():
    vector_path = Path('runs/wave_mode_switch_cma/best_vector.json')
    data = json.loads(vector_path.read_text())
    vector = np.asarray(data['vector'], dtype=np.float32)

    base_control = WaveFieldControlConfig()
    base_params = WaveModeSwitchParams()
    control_cfg, params = _config_from_vector(base_control, base_params, vector)

    controller = WaveFieldController(control_cfg, params.shape)
    simulator = WaveModeSwitchSimulator(params, controller)
    history = simulator.run(steps=1600, record_interval=50)
    final_metrics = history['metrics'][-1]

    assert final_metrics['phase'] == 'switched'
    assert final_metrics['mse'] <= max(data['metrics']['mean_switched_mse'] * 1.1, 0.5)
    assert np.isfinite(final_metrics['energy'])
    assert final_metrics['energy'] <= data['metrics']['mean_energy'] * 1.5
