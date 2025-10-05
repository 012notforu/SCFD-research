import numpy as np

from run.robustness_battery import (
    evaluate_flow_redundant,
    evaluate_heat_diffusion,
    evaluate_wave_mode_switch,
)


def _assert_scenarios(result: dict, expected: list[str]) -> None:
    assert "scenarios" in result
    scenarios = result["scenarios"]
    assert "baseline" in scenarios
    for name in expected:
        assert name in scenarios


def test_heat_diffusion_robustness_smoke() -> None:
    result = evaluate_heat_diffusion(
        steps=60,
        record_interval=20,
        seed=42,
        noise_scale=0.01,
        alpha_shift=1.1,
    )
    _assert_scenarios(result, ["noise", "alpha_shift"])
    summary = result["scenarios"]["baseline"]["summary"]
    assert summary["final_mse"] >= 0.0


def test_flow_redundant_robustness_smoke() -> None:
    result = evaluate_flow_redundant(
        steps=300,
        record_interval=100,
        seed=7,
        noise_scale=0.02,
        inflow_drop=0.9,
    )
    _assert_scenarios(result, ["actuator_noise", "inflow_drop"])
    summary = result["scenarios"]["baseline"]["summary"]
    assert np.isfinite(summary["final_throughput"])


def test_wave_mode_switch_robustness_smoke() -> None:
    result = evaluate_wave_mode_switch(
        steps=620,
        record_interval=200,
        seed=11,
        noise_scale=0.01,
        switch_delay=0,
    )
    _assert_scenarios(result, ["field_noise", "switch_delay"])
    summary = result["scenarios"]["baseline"]["summary"]
    assert np.isfinite(summary["final_mse"])
