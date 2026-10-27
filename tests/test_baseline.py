"""Utility-only baseline market: cost formula, signs, tariffs and output tables."""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from microgrid.baseline import (
    baseline_cost,
    community_baseline_cost,
    run_baseline,
    solve_baseline,
    summarize_baseline,
)
from microgrid.data import results_to_settlement_frame, results_to_trade_frame, validate_profiles
from microgrid.energy import compute_hour_states
from microgrid.models import (
    SETTLEMENT_CSV_COLUMNS,
    TRADE_CSV_COLUMNS,
    DataValidationError,
    Role,
    TariffSchedule,
)


@pytest.mark.parametrize(
    "s,r,b,a,expected",
    [
        (20, 0, 8, 3, -60),
        (0, 15, 8, 3, 120),
        (10, 0, 8, 3, -30),
        (0, 25, 8, 3, 200),
        (0, 0, 8, 3, 0),
        (0, 10, 0, 3, 0),
        (4, 0, 8, 0, 0),
    ],
)
def test_baseline_cost_formula(s, r, b, a, expected):
    assert baseline_cost(s, r, b, a) == pytest.approx(expected, abs=1e-12)


def test_golden_baseline_per_player(golden_state, golden_cfg, golden_expected):
    result = solve_baseline(golden_state, golden_cfg)
    for mg, exp in golden_expected.items():
        s = result.settlement_for(mg)
        assert s.baseline_cost_inr == pytest.approx(exp["baseline_cost"], abs=1e-9), mg
        assert s.grid_import_kwh == exp["deficit"] and s.grid_export_kwh == exp["surplus"]
        assert s.p2p_cost_inr == s.baseline_cost_inr and s.gain_inr == 0.0
    assert result.total_baseline_cost_inr == pytest.approx(230.0, abs=1e-9)
    assert result.total_cost_inr == pytest.approx(230.0, abs=1e-9)


def test_sign_convention(golden_state, golden_cfg):
    result = solve_baseline(golden_state, golden_cfg)
    for player in golden_state.players:
        cost = result.settlement_for(player.microgrid_id).baseline_cost_inr
        if player.role is Role.SELLER:
            assert cost < 0, "sellers earn export income, so their signed cost is negative"
        else:
            assert cost > 0


def test_baseline_result_structure_and_money_identity(golden_state, golden_cfg):
    result = solve_baseline(golden_state, golden_cfg)
    assert result.mechanism == "baseline"
    assert result.trades == () and result.total_p2p_volume_kwh == 0.0
    assert result.diagnostics.status == "success"
    for s in result.settlements:
        assert s.p2p_import_kwh == s.p2p_export_kwh == 0.0
        assert s.grid_purchase_inr == pytest.approx(8.0 * s.grid_import_kwh)
        assert s.grid_export_revenue_inr == pytest.approx(3.0 * s.grid_export_kwh)
        recomputed = s.grid_purchase_inr - s.grid_export_revenue_inr + s.p2p_purchase_inr - s.p2p_sales_inr + s.fees_inr
        assert s.p2p_cost_inr == pytest.approx(recomputed, abs=1e-12)


def test_fees_do_not_affect_baseline(golden_state, golden_cfg):
    with_fees = dataclasses.replace(golden_cfg, seller_fee_inr_per_kwh=1.0, buyer_fee_inr_per_kwh=1.0)
    assert solve_baseline(golden_state, with_fees).total_baseline_cost_inr == pytest.approx(230.0)


def test_hourly_tariff_table_is_used(golden_state, golden_cfg):
    retail = [8.0] * 24
    export = [3.0] * 24
    retail[12], export[12] = 10.0, 2.0  # only the golden hour changes
    result = solve_baseline(golden_state, golden_cfg, TariffSchedule(tuple(retail), tuple(export)))
    assert result.settlement_for("MG2").baseline_cost_inr == pytest.approx(150.0)
    assert result.settlement_for("MG1").baseline_cost_inr == pytest.approx(-40.0)
    assert result.total_baseline_cost_inr == pytest.approx(10.0 * 40 - 2.0 * 30)


def test_cost_scales_linearly_with_prices(golden_state, golden_cfg):
    doubled = solve_baseline(golden_state, golden_cfg, TariffSchedule.flat(16.0, 6.0))
    assert doubled.total_baseline_cost_inr == pytest.approx(460.0)


def test_balanced_player_costs_nothing(golden_raw, golden_cfg):
    golden_raw.loc[0, ["generation_kwh", "demand_kwh"]] = [12.0, 12.0]
    state = compute_hour_states(validate_profiles(golden_raw, golden_cfg), golden_cfg.tol)[0]
    assert state.player("MG1").role is Role.BALANCED
    s = solve_baseline(state, golden_cfg).settlement_for("MG1")
    assert s.baseline_cost_inr == 0.0 and s.grid_import_kwh == 0.0 and s.grid_export_kwh == 0.0


def test_24h_baseline_matches_independent_formula(sample_profiles, default_cfg):
    results = run_baseline(sample_profiles, default_cfg)
    assert [(r.day, r.hour) for r in results] == [(0, h) for h in range(24)]
    g, d = sample_profiles.generation_kwh, sample_profiles.demand_kwh
    expected = (8.0 * (d - g).clip(lower=0) - 3.0 * (g - d).clip(lower=0)).sum()
    assert community_baseline_cost(results) == pytest.approx(expected, rel=1e-12)


def test_output_tables_follow_csv_contract(sample_profiles, default_cfg):
    results = run_baseline(sample_profiles, default_cfg)
    settlement = results_to_settlement_frame(results)
    assert tuple(settlement.columns) == SETTLEMENT_CSV_COLUMNS
    assert len(settlement) == 96
    assert set(settlement.mechanism) == {"baseline"}
    assert set(settlement.scenario_id) == {"four_microgrids_default"}
    assert (settlement.gain_inr == 0).all()
    trades = results_to_trade_frame(results)
    assert tuple(trades.columns) == TRADE_CSV_COLUMNS and trades.empty


def test_golden_settlement_row_matches_blueprint_example(golden_profiles, golden_cfg):
    row = results_to_settlement_frame(run_baseline(golden_profiles, golden_cfg)).iloc[0]
    assert (row.day, row.hour, row.microgrid_id) == (0, 12, "MG1")
    assert (row.grid_import_kwh, row.grid_export_kwh, row.p2p_import_kwh, row.p2p_export_kwh) == (0, 20, 0, 0)
    assert row.baseline_cost_inr == -60


def test_summary_totals_match_community_cost(sample_profiles, default_cfg):
    results = run_baseline(sample_profiles, default_cfg)
    summary = summarize_baseline(results)
    assert summary.microgrid_id.tolist() == ["MG1", "MG2", "MG3", "MG4"]
    assert summary.baseline_cost_inr.sum() == pytest.approx(community_baseline_cost(results), rel=1e-12)
    assert summarize_baseline([]).empty


def test_run_baseline_validates_its_input(golden_raw, golden_cfg):
    negative = golden_raw.copy()
    negative.loc[0, "generation_kwh"] = -1.0
    with pytest.raises(DataValidationError, match="negative energy"):
        run_baseline(negative, golden_cfg)
    with pytest.raises(DataValidationError, match="missing 1 record"):
        run_baseline(golden_raw.drop(index=3), golden_cfg)
