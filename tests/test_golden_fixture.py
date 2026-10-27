"""Golden single-hour fixture (blueprint sections 4 and 7): regression anchor.

Baseline (Phase 1): community net cost INR 230.
Cooperative (Phase 2): trades (MG1->MG2, 15 kWh), (MG1->MG4, 5 kWh),
(MG3->MG4, 10 kWh) at INR 5.50; P2P community cost INR 80; gain INR 150;
Jain fairness 0.94736842. With fees 1 + 1: gain INR 90, fees INR 60. With
fees summing to at least 5, no trade happens.
Competitive (Phase 3): unique pure NE (MG1 7.50, MG3 7.50) among 100
profiles; same trades at INR 7.50; cost INR 80, gain INR 150, split
90 / 7.5 / 45 / 7.5 (Jain 0.54945055). These values were computed by
code and verified by exhaustive deviation checks; they are not taken from
the blueprint, which gives no competitive golden numbers.
"""

from __future__ import annotations

import dataclasses

import pytest

from microgrid.baseline import community_baseline_cost, run_baseline
from microgrid.competitive import solve_competitive
from microgrid.cooperative import run_cooperative, solve_cooperative
from microgrid.data import results_to_settlement_frame, results_to_trade_frame
from microgrid.metrics import jain_fairness

#            id,   G,  D,  S,  R
GOLDEN = [("MG1", 40, 20, 20, 0), ("MG2", 10, 25, 0, 15), ("MG3", 25, 15, 10, 0), ("MG4", 5, 30, 0, 25)]


def test_fixture_file_matches_blueprint_table(golden_profiles):
    assert len(golden_profiles) == 4
    assert set(golden_profiles.day) == {0} and set(golden_profiles.hour) == {12}
    rows = list(golden_profiles[["microgrid_id", "generation_kwh", "demand_kwh"]].itertuples(index=False))
    assert [(r.microgrid_id, r.generation_kwh, r.demand_kwh) for r in rows] == [(m, g, d) for m, g, d, _, _ in GOLDEN]


@pytest.mark.parametrize("mg,g,d,s,r", GOLDEN)
def test_surplus_and_deficit_per_microgrid(golden_state, mg, g, d, s, r):
    player = golden_state.player(mg)
    assert (player.generation_kwh, player.demand_kwh) == (g, d)
    assert player.surplus_kwh == s
    assert player.deficit_kwh == r


def test_totals_row(golden_state):
    assert golden_state.total_generation_kwh == 80
    assert golden_state.total_demand_kwh == 90
    assert golden_state.total_surplus_kwh == 30  # total available surplus
    assert golden_state.total_deficit_kwh == 40  # total deficit


def test_baseline_net_costs(golden_profiles, golden_cfg, golden_expected):
    results = run_baseline(golden_profiles, golden_cfg)
    assert len(results) == 1
    for mg, exp in golden_expected.items():
        assert results[0].settlement_for(mg).baseline_cost_inr == pytest.approx(exp["baseline_cost"], abs=1e-9)
    assert community_baseline_cost(results) == pytest.approx(230.0, abs=1e-9)


def test_costs_are_aggregated_with_signs(golden_profiles, golden_cfg):
    result = run_baseline(golden_profiles, golden_cfg)[0]
    signed = sum(s.baseline_cost_inr for s in result.settlements)
    absolute = sum(abs(s.baseline_cost_inr) for s in result.settlements)
    assert signed == pytest.approx(230.0)
    assert absolute == pytest.approx(410.0)  # the wrong answer if signs were dropped


# --- cooperative market (Phase 2) -------------------------------------------

#            baseline  cooperative  gain
GOLDEN_COOPERATIVE = {
    "MG1": (-60.0, -110.0, 50.0),
    "MG2": (120.0, 82.5, 37.5),
    "MG3": (-30.0, -55.0, 25.0),
    "MG4": (200.0, 162.5, 37.5),
}


def _with_fees(cfg, seller_fee, buyer_fee):
    return dataclasses.replace(cfg, seller_fee_inr_per_kwh=seller_fee, buyer_fee_inr_per_kwh=buyer_fee)


def test_cooperative_trades(golden_state, golden_cfg):  # T02, T03
    result = solve_cooperative(golden_state, golden_cfg)
    rows = [(t.seller_id, t.buyer_id, t.quantity_kwh, t.price_inr_per_kwh, t.payment_inr) for t in result.trades]
    assert rows == [
        ("MG1", "MG2", 15.0, 5.5, 82.5),
        ("MG1", "MG4", 5.0, 5.5, 27.5),
        ("MG3", "MG4", 10.0, 5.5, 55.0),
    ]
    assert result.total_p2p_volume_kwh == 30.0
    assert result.total_grid_import_kwh == 10.0  # residual utility imports, all by MG4
    assert result.settlement_for("MG4").grid_import_kwh == 10.0


def test_cooperative_costs_and_gains(golden_state, golden_cfg):  # T01
    result = solve_cooperative(golden_state, golden_cfg)
    for mg, (base, cost, gain) in GOLDEN_COOPERATIVE.items():
        s = result.settlement_for(mg)
        assert (s.baseline_cost_inr, s.p2p_cost_inr, s.gain_inr) == pytest.approx((base, cost, gain), abs=1e-9), mg
    assert result.total_baseline_cost_inr == pytest.approx(230.0)
    assert result.total_cost_inr == pytest.approx(80.0)
    assert result.total_gain_inr == pytest.approx(150.0)  # = (8 - 3) * 30
    assert 100 * result.total_gain_inr / result.total_baseline_cost_inr == pytest.approx(65.21739, abs=1e-5)
    assert result.diagnostics.status == "success"
    assert result.diagnostics.optimal_welfare_inr == pytest.approx(150.0)


def test_cooperative_jain_fairness(golden_state, golden_cfg):
    gains = [s.gain_inr for s in solve_cooperative(golden_state, golden_cfg).settlements]
    assert jain_fairness(gains) == pytest.approx(0.94736842, abs=1e-8)


def test_cooperative_csv_rows_match_blueprint_examples(golden_profiles, golden_cfg):
    results = run_cooperative(golden_profiles, golden_cfg)
    trade = results_to_trade_frame(results).iloc[0]
    assert list(trade)[1:] == ["cooperative", 0, 12, "MG1", "MG2", 15.0, 5.5, 0.0, 0.0]
    settlement = results_to_settlement_frame(results).iloc[0]
    assert list(settlement)[1:] == ["cooperative", 0, 12, "MG1", 0.0, 0.0, 0.0, 20.0, -60.0, -110.0, 50.0]


def test_fee_fixture(golden_state, golden_cfg):  # T04
    result = solve_cooperative(golden_state, _with_fees(golden_cfg, 1.0, 1.0))
    assert [(t.seller_id, t.buyer_id, t.quantity_kwh) for t in result.trades] == [
        ("MG1", "MG2", 15.0), ("MG1", "MG4", 5.0), ("MG3", "MG4", 10.0)
    ]
    for t in result.trades:
        assert t.price_inr_per_kwh == pytest.approx(5.5)  # gross price unchanged
        assert t.seller_net_price_inr_per_kwh == pytest.approx(4.5)
        assert t.buyer_delivered_price_inr_per_kwh == pytest.approx(6.5)
    assert result.total_gain_inr == pytest.approx(90.0)  # 30 * (8 - 3 - 1 - 1)
    assert result.total_fee_revenue_inr == pytest.approx(60.0)  # 30 * 2
    assert result.total_cost_inr == pytest.approx(140.0)


@pytest.mark.parametrize("seller_fee,buyer_fee", [(2.5, 2.5), (3.0, 2.0), (5.0, 0.0), (0.0, 5.0), (3.0, 3.0)])
def test_trading_stops_when_fees_absorb_the_spread(golden_state, golden_cfg, seller_fee, buyer_fee):  # T05
    result = solve_cooperative(golden_state, _with_fees(golden_cfg, seller_fee, buyer_fee))
    assert result.trades == ()
    assert result.diagnostics.status == "no_trade"
    assert "not positive" in result.diagnostics.message
    assert all(s.gain_inr == 0.0 and s.p2p_cost_inr == s.baseline_cost_inr for s in result.settlements)
    assert result.total_cost_inr == pytest.approx(230.0)
    gains = [s.gain_inr for s in result.settlements]
    assert jain_fairness(gains) is None  # no division by zero


def test_trading_resumes_just_below_the_threshold(golden_state, golden_cfg):
    result = solve_cooperative(golden_state, _with_fees(golden_cfg, 2.49, 2.5))
    assert result.total_p2p_volume_kwh == pytest.approx(30.0)
    assert result.total_gain_inr == pytest.approx(30 * 0.01)


# --- competitive market (Phase 3) -------------------------------------------

#            baseline  competitive  gain
GOLDEN_COMPETITIVE = {
    "MG1": (-60.0, -150.0, 90.0),
    "MG2": (120.0, 112.5, 7.5),
    "MG3": (-30.0, -75.0, 45.0),
    "MG4": (200.0, 192.5, 7.5),
}


def test_competitive_equilibrium(golden_state, golden_cfg):
    result = solve_competitive(golden_state, golden_cfg)
    d = result.diagnostics
    assert d.status == "success" and d.pure_ne_found is True
    assert d.equilibrium_count == 1 and d.examined_profiles == 100  # 10 prices x 10 prices
    assert d.selected_profile == (("MG1", 7.5), ("MG3", 7.5))
    assert d.max_regret_inr == pytest.approx(0.0, abs=1e-12)


def test_competitive_trades_costs_and_gains(golden_state, golden_cfg):
    result = solve_competitive(golden_state, golden_cfg)
    rows = [(t.seller_id, t.buyer_id, t.quantity_kwh, t.price_inr_per_kwh) for t in result.trades]
    assert rows == [("MG1", "MG2", 15.0, 7.5), ("MG1", "MG4", 5.0, 7.5), ("MG3", "MG4", 10.0, 7.5)]
    for mg, (base, cost, gain) in GOLDEN_COMPETITIVE.items():
        s = result.settlement_for(mg)
        assert (s.baseline_cost_inr, s.p2p_cost_inr, s.gain_inr) == pytest.approx((base, cost, gain), abs=1e-9), mg
    # same volume and total gain as the cooperative market, very different distribution
    assert result.total_cost_inr == pytest.approx(80.0)
    assert result.total_gain_inr == pytest.approx(150.0)
    assert jain_fairness([s.gain_inr for s in result.settlements]) == pytest.approx(22500 / 40950)


def test_competitive_fee_fixture(golden_state, golden_cfg):
    result = solve_competitive(golden_state, _with_fees(golden_cfg, 1.0, 1.0))
    assert result.diagnostics.selected_profile == (("MG1", 6.5), ("MG3", 6.5))
    assert result.total_gain_inr == pytest.approx(90.0)
    assert result.total_fee_revenue_inr == pytest.approx(60.0)


def test_competitive_trading_stops_when_fees_absorb_the_spread(golden_state, golden_cfg):
    result = solve_competitive(golden_state, _with_fees(golden_cfg, 2.5, 2.5))
    assert result.trades == () and result.diagnostics.status == "no_trade"
    assert result.total_cost_inr == pytest.approx(230.0)
