"""Edge cases: night hours, one-sided markets, balanced players, immutability (T06, T07, T13, T16)."""

from __future__ import annotations

import pandas as pd
import pytest

from microgrid.baseline import run_baseline, solve_baseline
from microgrid.data import validate_profiles
from microgrid.energy import add_surplus_deficit_columns, compute_hour_state, compute_hour_states
from microgrid.models import Role


def _single_hour(cfg, generation, demand, hour=0):
    frame = pd.DataFrame(
        {
            "day": 0,
            "hour": hour,
            "microgrid_id": ["MG1", "MG2", "MG3", "MG4"],
            "generation_kwh": generation,
            "demand_kwh": demand,
        }
    )
    return compute_hour_states(validate_profiles(frame, cfg), cfg.tol)[0]


def test_night_hour_has_no_sellers(sample_profiles, default_cfg):  # T06
    night = compute_hour_states(sample_profiles, default_cfg.tol)[0]
    assert night.hour == 0
    assert night.sellers == ()
    assert all(p.generation_kwh == 0.0 for p in night.players)
    assert all(p.role is Role.BUYER for p in night.players)
    result = solve_baseline(night, default_cfg)
    assert result.total_grid_export_kwh == 0.0
    assert all(s.baseline_cost_inr >= 0 for s in result.settlements)
    assert result.total_baseline_cost_inr == pytest.approx(8.0 * night.total_deficit_kwh)


def test_market_with_no_sellers(golden_cfg):  # T07
    state = _single_hour(golden_cfg, [0, 0, 0, 0], [5, 6, 7, 8])
    assert state.sellers == () and state.buyers == ("MG1", "MG2", "MG3", "MG4")
    result = solve_baseline(state, golden_cfg)
    assert result.trades == ()
    assert result.total_baseline_cost_inr == pytest.approx(8.0 * 26)


def test_market_with_no_buyers(golden_cfg):  # T07
    state = _single_hour(golden_cfg, [10, 10, 10, 10], [1, 2, 3, 4])
    assert state.buyers == ()
    result = solve_baseline(state, golden_cfg)
    assert result.total_baseline_cost_inr == pytest.approx(-3.0 * (9 + 8 + 7 + 6))


def test_all_balanced_and_all_zero(golden_cfg):
    for values in ([5, 6, 7, 8], [0, 0, 0, 0]):
        state = _single_hour(golden_cfg, values, values)
        assert all(p.role is Role.BALANCED for p in state.players)
        assert solve_baseline(state, golden_cfg).total_baseline_cost_inr == 0.0


def test_imbalance_below_tolerance_is_balanced(golden_cfg):
    state = _single_hour(golden_cfg, [10 + 1e-12, 1, 1, 1], [10, 1, 1, 1])
    assert state.player("MG1").role is Role.BALANCED
    assert solve_baseline(state, golden_cfg).total_baseline_cost_inr == pytest.approx(0.0, abs=1e-9)


def test_large_values_stay_exact(golden_cfg):
    state = _single_hour(golden_cfg, [1e6, 0, 0, 0], [0, 0, 0, 0])
    assert solve_baseline(state, golden_cfg).total_baseline_cost_inr == -3e6


def test_inputs_are_not_mutated(sample_profiles, default_cfg):  # T16
    snapshot = sample_profiles.copy(deep=True)
    validate_profiles(sample_profiles, default_cfg)
    add_surplus_deficit_columns(sample_profiles)
    compute_hour_states(sample_profiles, default_cfg.tol)
    run_baseline(sample_profiles, default_cfg)
    pd.testing.assert_frame_equal(sample_profiles, snapshot)


def test_repeated_runs_are_identical(sample_profiles, default_cfg):  # T13
    assert run_baseline(sample_profiles, default_cfg) == run_baseline(sample_profiles, default_cfg)


def test_hour_state_rejects_rows_from_several_hours(golden_raw):
    two_hours = pd.concat([golden_raw, golden_raw.assign(hour=13)], ignore_index=True)
    with pytest.raises(ValueError, match="more than one interval"):
        compute_hour_state(two_hours)
