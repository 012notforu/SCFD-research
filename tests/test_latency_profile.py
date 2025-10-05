from run.latency_profile import (
    profile_flow_redundant,
    profile_heat_diffusion,
    profile_wave_mode_switch,
)



def _check_common_keys(payload: dict) -> None:
    for key in ("avg_step_ms", "std_step_ms", "max_step_ms", "vector_path", "vector_metrics"):
        assert key in payload


def test_profile_heat_diffusion_smoke() -> None:
    result = profile_heat_diffusion(steps=80, warmup=10, seed=1)
    _check_common_keys(result)
    assert "mean_abs_delta" in result
    assert result["steps_measured"] >= 0


def test_profile_flow_redundant_smoke() -> None:
    result = profile_flow_redundant(steps=120, warmup=20, seed=3)
    _check_common_keys(result)
    assert "final_throughput" in result
    assert "final_budget_util" in result


def test_profile_wave_mode_switch_smoke() -> None:
    result = profile_wave_mode_switch(steps=200, warmup=20, seed=5)
    _check_common_keys(result)
    assert "boundary_mean_abs_delta" in result
    assert "final_phase" in result