"""Cooperative market: LP allocation, tie rule, solver failures and invariants (T07-T10, T13, T16).

The golden-fixture numbers (80 / 150 / 30 kWh, fee fixture) are in
test_golden_fixture.py.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import OptimizeResult

import microgrid.cooperative as cooperative
from microgrid.baseline import community_baseline_cost, run_baseline
from microgrid.cooperative import (
    Edge,
    allocate_max_welfare,
    eligible_edges,
    greedy_allocation,
    run_cooperative,
    solve_cooperative,
)
from microgrid.data import generate_profiles, results_to_settlement_frame, results_to_trade_frame, validate_profiles
from microgrid.energy import compute_hour_states
from microgrid.models import SETTLEMENT_CSV_COLUMNS, TRADE_CSV_COLUMNS, ScenarioConfig, SolverError, TariffSchedule

KEYS = ["day", "hour", "microgrid_id"]


def _single_hour(cfg, generation, demand):
    frame = pd.DataFrame(
        {"day": 0, "hour": 12, "microgrid_id": ["MG1", "MG2", "MG3", "MG4"],
         "generation_kwh": generation, "demand_kwh": demand}
    )
    return compute_hour_states(validate_profiles(frame, cfg), cfg.tol)[0]


@pytest.fixture
def sample_results(sample_profiles, default_cfg):
    return run_cooperative(sample_profiles, default_cfg)


# --- mechanism behaviour ----------------------------------------------------


@pytest.mark.parametrize("alpha", [0.2, 0.5, 0.8])
def test_alpha_redistributes_without_changing_total_gain(golden_state, golden_cfg, alpha):
    result = solve_cooperative(golden_state, dataclasses.replace(golden_cfg, bargaining_seller_weight=alpha))
    assert result.total_gain_inr == pytest.approx(150.0)
    assert all(t.price_inr_per_kwh == pytest.approx(3.0 + alpha * 5.0) for t in result.trades)
    sellers_gain = result.settlement_for("MG1").gain_inr + result.settlement_for("MG3").gain_inr
    assert sellers_gain == pytest.approx(alpha * 150.0)


def test_hourly_tariffs_change_price_and_gain(golden_state, golden_cfg):
    retail, export = [8.0] * 24, [3.0] * 24
    retail[12], export[12] = 10.0, 2.0
    result = solve_cooperative(golden_state, golden_cfg, TariffSchedule(tuple(retail), tuple(export)))
    assert result.trades and all(t.price_inr_per_kwh == pytest.approx(6.0) for t in result.trades)
    assert result.total_gain_inr == pytest.approx(30 * 8.0)


def test_scarce_demand_leaves_seller_surplus_for_export(golden_cfg):
    state = _single_hour(golden_cfg, [40, 10, 25, 5], [20, 12, 15, 8])  # deficits 2 and 3
    result = solve_cooperative(state, golden_cfg)
    assert [(t.seller_id, t.buyer_id, t.quantity_kwh) for t in result.trades] == [
        ("MG1", "MG2", pytest.approx(2.0)),
        ("MG1", "MG4", pytest.approx(3.0)),
    ]
    assert result.settlement_for("MG1").grid_export_kwh == pytest.approx(15.0)
    assert result.settlement_for("MG3").grid_export_kwh == pytest.approx(10.0)
    assert result.total_gain_inr == pytest.approx(5 * 5.0)


@pytest.mark.parametrize(
    "generation,demand", [([0, 0, 0, 0], [5, 6, 7, 8]), ([10, 10, 10, 10], [1, 2, 3, 4]), ([5, 5, 5, 5], [5, 5, 5, 5])]
)
def test_no_sellers_or_no_buyers_gives_empty_balanced_settlement(golden_cfg, generation, demand):  # T07
    state = _single_hour(golden_cfg, generation, demand)
    result = solve_cooperative(state, golden_cfg)
    assert result.trades == ()
    assert result.diagnostics.status == "no_trade"
    assert "no sellers or no buyers" in result.diagnostics.message
    assert all(s.p2p_cost_inr == s.baseline_cost_inr and s.gain_inr == 0.0 for s in result.settlements)


# --- 24-hour invariants -----------------------------------------------------


def test_capacities_are_respected_every_hour(sample_results, sample_profiles, default_cfg):  # T08
    for state, result in zip(compute_hour_states(sample_profiles, default_cfg.tol), sample_results):
        for p in state.players:
            sold = math.fsum(t.quantity_kwh for t in result.trades if t.seller_id == p.microgrid_id)
            bought = math.fsum(t.quantity_kwh for t in result.trades if t.buyer_id == p.microgrid_id)
            assert sold <= p.surplus_kwh + 1e-9
            assert bought <= p.deficit_kwh + 1e-9
        assert result.total_p2p_volume_kwh == pytest.approx(
            min(state.total_surplus_kwh, state.total_deficit_kwh), abs=1e-7
        )


def test_payments_cancel_every_hour(sample_results):  # T09
    for result in sample_results:
        purchases = math.fsum(s.p2p_purchase_inr for s in result.settlements)
        sales = math.fsum(s.p2p_sales_inr for s in result.settlements)
        assert purchases == pytest.approx(sales, abs=1e-9)


def test_energy_identity_every_player_and_hour(sample_results, sample_profiles):  # T10
    flows = sample_profiles.merge(results_to_settlement_frame(sample_results), on=KEYS, validate="one_to_one")
    supply = flows.generation_kwh + flows.grid_import_kwh + flows.p2p_import_kwh
    use = flows.demand_kwh + flows.grid_export_kwh + flows.p2p_export_kwh
    np.testing.assert_allclose(supply, use, rtol=0, atol=1e-9)
    assert not ((flows.grid_import_kwh > 1e-9) & (flows.grid_export_kwh > 1e-9)).any()
    assert not ((flows.p2p_import_kwh > 1e-9) & (flows.p2p_export_kwh > 1e-9)).any()


def test_24h_gain_matches_independent_formula(sample_results, sample_profiles, default_cfg):
    states = compute_hour_states(sample_profiles, default_cfg.tol)
    expected_gain = math.fsum(5.0 * min(s.total_surplus_kwh, s.total_deficit_kwh) for s in states)
    gain = math.fsum(r.total_gain_inr for r in sample_results)
    assert gain == pytest.approx(expected_gain, rel=1e-9)
    baseline = community_baseline_cost(run_baseline(sample_profiles, default_cfg))
    assert math.fsum(r.total_cost_inr for r in sample_results) == pytest.approx(baseline - gain, rel=1e-9)


def test_participation_is_voluntary(sample_results):
    for result in sample_results:
        assert all(s.gain_inr >= -1e-9 for s in result.settlements)
        for t in result.trades:
            assert t.price_inr_per_kwh == pytest.approx(5.5)
            assert 3.0 < t.seller_net_price_inr_per_kwh and t.buyer_delivered_price_inr_per_kwh < 8.0


def test_lp_optimum_equals_realised_gain(sample_results):
    for result in sample_results:
        assert result.diagnostics.optimal_welfare_inr == pytest.approx(result.total_gain_inr, abs=1e-6)
        if not result.trades:
            assert result.diagnostics.status == "no_trade"


def test_night_hours_have_no_trades(sample_results):
    for result in sample_results[:6]:
        assert result.trades == () and result.diagnostics.status == "no_trade"


def test_output_tables_follow_csv_contract(sample_results):
    settlement = results_to_settlement_frame(sample_results)
    trades = results_to_trade_frame(sample_results)
    assert tuple(settlement.columns) == SETTLEMENT_CSV_COLUMNS and len(settlement) == 96
    assert tuple(trades.columns) == TRADE_CSV_COLUMNS and not trades.empty
    assert set(settlement.mechanism) == {"cooperative"} and set(trades.mechanism) == {"cooperative"}


def test_deterministic_and_inputs_unchanged(sample_profiles, default_cfg):  # T13, T16
    snapshot = sample_profiles.copy(deep=True)
    assert run_cooperative(sample_profiles, default_cfg) == run_cooperative(sample_profiles, default_cfg)
    pd.testing.assert_frame_equal(sample_profiles, snapshot)


def test_six_microgrid_market():
    cfg = ScenarioConfig(microgrid_ids=tuple(f"MG{i}" for i in range(1, 7)))
    profiles = generate_profiles(cfg)
    results = run_cooperative(profiles, cfg)
    states = compute_hour_states(profiles, cfg.tol)
    expected = math.fsum(5.0 * min(s.total_surplus_kwh, s.total_deficit_kwh) for s in states)
    assert math.fsum(r.total_gain_inr for r in results) == pytest.approx(expected, rel=1e-9)


# --- LP core: allocation and tie rule ---------------------------------------


def test_golden_edges_lp_and_greedy_agree(golden_state, golden_cfg):
    edges = eligible_edges(golden_state, golden_cfg)
    assert [(e.seller_id, e.buyer_id) for e in edges] == [("MG1", "MG2"), ("MG1", "MG4"), ("MG3", "MG2"), ("MG3", "MG4")]
    assert {e.unit_surplus_inr_per_kwh for e in edges} == {5.0}
    supply, demand = {"MG1": 20.0, "MG3": 10.0}, {"MG2": 15.0, "MG4": 25.0}
    assert greedy_allocation(supply, demand, edges) == (15.0, 5.0, 0.0, 10.0)
    allocation = allocate_max_welfare(supply, demand, edges)
    assert allocation.quantities_kwh == pytest.approx((15.0, 5.0, 0.0, 10.0), abs=1e-9)
    assert allocation.optimal_welfare_inr == pytest.approx(150.0)
    assert allocation.lp_solves == 5
    assert "equals the sorted greedy allocation" in allocation.cross_check
    # the zero edge MG3->MG2 does not become a phantom trade
    pairs = [(t.seller_id, t.buyer_id) for t in solve_cooperative(golden_state, golden_cfg).trades]
    assert ("MG3", "MG2") not in pairs


def test_lp_beats_greedy_when_unit_surplus_differs():
    edges = [Edge("A", "X", 1.0), Edge("A", "Y", 2.0)]
    allocation = allocate_max_welfare({"A": 10.0}, {"X": 10.0, "Y": 10.0}, edges)
    assert allocation.quantities_kwh == pytest.approx((0.0, 10.0))
    assert allocation.optimal_welfare_inr == pytest.approx(20.0)
    assert greedy_allocation({"A": 10.0}, {"X": 10.0, "Y": 10.0}, edges) == (10.0, 0.0)  # greedy only gets 10
    assert "does not exceed the LP optimum" in allocation.cross_check


def test_missing_dual_values_raise_solver_error(monkeypatch):
    real_linprog = cooperative.linprog

    def no_duals(c, **kwargs):
        result = real_linprog(c, **kwargs)
        del result["ineqlin"]
        return result

    monkeypatch.setattr(cooperative, "linprog", no_duals)
    with pytest.raises(SolverError, match="no dual values"):
        allocate_max_welfare({"A": 5.0}, {"X": 5.0}, [Edge("A", "X", 1.0)])


def test_tie_rule_is_lexicographic_among_optimal_allocations():
    edges = [Edge("A", "X", 2.0), Edge("A", "Y", 2.0), Edge("B", "X", 1.0), Edge("B", "Y", 1.0)]
    allocation = allocate_max_welfare({"A": 10.0, "B": 10.0}, {"X": 10.0, "Y": 10.0}, edges)
    assert allocation.optimal_welfare_inr == pytest.approx(30.0)
    assert allocation.quantities_kwh == pytest.approx((10.0, 0.0, 0.0, 10.0), abs=1e-9)


def test_lp_equals_greedy_on_random_uniform_instances():
    rng = np.random.default_rng(7)
    for _ in range(25):
        supply = {f"S{i}": round(float(rng.uniform(0.5, 40.0)), 4) for i in range(int(rng.integers(1, 4)))}
        demand = {f"B{i}": round(float(rng.uniform(0.5, 40.0)), 4) for i in range(int(rng.integers(1, 4)))}
        edges = [Edge(s, b, 5.0) for s in supply for b in demand]
        allocation = allocate_max_welfare(supply, demand, edges)
        np.testing.assert_allclose(allocation.quantities_kwh, greedy_allocation(supply, demand, edges), atol=1e-7)
        assert allocation.total_kwh == pytest.approx(min(sum(supply.values()), sum(demand.values())), abs=1e-7)


def test_lp_is_at_least_as_good_as_greedy_on_random_non_uniform_instances():
    rng = np.random.default_rng(11)
    for _ in range(25):
        supply = {f"S{i}": float(rng.uniform(1.0, 30.0)) for i in range(3)}
        demand = {f"B{i}": float(rng.uniform(1.0, 30.0)) for i in range(3)}
        edges = [Edge(s, b, float(rng.uniform(0.5, 5.0))) for s in supply for b in demand]
        allocation = allocate_max_welfare(supply, demand, edges)
        greedy = greedy_allocation(supply, demand, edges)
        greedy_welfare = sum(e.unit_surplus_inr_per_kwh * q for e, q in zip(allocation.edges, greedy))
        assert allocation.welfare_inr >= greedy_welfare - 1e-7
        assert allocation.welfare_inr == pytest.approx(allocation.optimal_welfare_inr, rel=1e-7)
        for s, cap in supply.items():
            assert sum(q for e, q in zip(allocation.edges, allocation.quantities_kwh) if e.seller_id == s) <= cap + 1e-9
        for b, cap in demand.items():
            assert sum(q for e, q in zip(allocation.edges, allocation.quantities_kwh) if e.buyer_id == b) <= cap + 1e-9


def test_allocation_input_validation():
    with pytest.raises(ValueError, match="duplicate edges"):
        allocate_max_welfare({"A": 1.0}, {"X": 1.0}, [Edge("A", "X", 1.0), Edge("A", "X", 1.0)])
    with pytest.raises(ValueError, match="unit surplus > tol"):
        allocate_max_welfare({"A": 1.0}, {"X": 1.0}, [Edge("A", "X", 0.0)])
    with pytest.raises(ValueError, match="no matching supply"):
        allocate_max_welfare({}, {"X": 1.0}, [Edge("A", "X", 1.0)])
    with pytest.raises(ValueError, match="finite and >= 0"):
        allocate_max_welfare({"A": -1.0}, {"X": 1.0}, [Edge("A", "X", 1.0)])
    empty = allocate_max_welfare({}, {}, [])
    assert empty.quantities_kwh == () and empty.optimal_welfare_inr == 0.0 and empty.lp_solves == 0


# --- solver failures are never reported as results ---------------------------


def test_failed_lp_raises_solver_error(monkeypatch, golden_state, golden_cfg):
    def infeasible(c, **kwargs):
        return OptimizeResult(x=None, status=2, success=False, message="The problem is infeasible.", fun=None)

    monkeypatch.setattr(cooperative, "linprog", infeasible)
    with pytest.raises(SolverError, match="status 2"):
        solve_cooperative(golden_state, golden_cfg)


def test_failure_during_tie_break_raises_solver_error(monkeypatch, golden_state, golden_cfg):
    real_linprog = cooperative.linprog
    calls = {"n": 0}

    def flaky(c, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return real_linprog(c, **kwargs)
        return OptimizeResult(x=None, status=4, success=False, message="Numerical difficulties.", fun=None)

    monkeypatch.setattr(cooperative, "linprog", flaky)
    with pytest.raises(SolverError, match="tie-break"):
        solve_cooperative(golden_state, golden_cfg)


def test_wrong_lp_answer_is_caught_by_the_greedy_cross_check(monkeypatch, golden_state, golden_cfg):
    def claims_success_with_zeros(c, **kwargs):
        rows = 0 if kwargs.get("A_ub") is None else kwargs["A_ub"].shape[0]
        return OptimizeResult(
            x=np.zeros(len(c)), status=0, success=True, message="fake", fun=0.0,
            ineqlin=OptimizeResult(marginals=np.zeros(rows)), lower=OptimizeResult(marginals=np.zeros(len(c))),
        )

    monkeypatch.setattr(cooperative, "linprog", claims_success_with_zeros)
    with pytest.raises(SolverError, match="differs from the sorted greedy allocation"):
        solve_cooperative(golden_state, golden_cfg)
