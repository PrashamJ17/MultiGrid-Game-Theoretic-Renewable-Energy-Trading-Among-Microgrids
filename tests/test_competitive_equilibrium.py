"""Competitive seller price game: strategy sets, buyer rule and exhaustive pure-NE verification (T11, T12).

Equilibria are cross-checked against an independent, deliberately naive
reference written directly from the blueprint text: buyer rule, payoff
definition, and an explicit loop over every unilateral deviation. It shares
no code with ``microgrid.competitive``.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import warnings

import pandas as pd
import pytest

from microgrid.competitive import (
    STATUS_CAP_EXCEEDED,
    STATUS_NO_PURE_NE,
    allocate_buyers,
    analyze_price_game,
    price_grid,
    run_competitive,
    solve_competitive,
)
from microgrid.cooperative import run_cooperative
from microgrid.data import generate_profiles, validate_profiles
from microgrid.energy import compute_hour_states
from microgrid.models import CombinatorialWarning, ScenarioConfig

DEFAULT_GRID = (3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5)


def _game(generation, demand, **overrides):
    ids = tuple(f"MG{i}" for i in range(1, len(generation) + 1))
    cfg = ScenarioConfig(scenario_id="game", microgrid_ids=ids, require_full_horizon=False, **overrides)
    frame = pd.DataFrame(
        {"day": 0, "hour": 12, "microgrid_id": list(ids), "generation_kwh": generation, "demand_kwh": demand}
    )
    return cfg, compute_hour_states(validate_profiles(frame, cfg), cfg.tol)[0]


# --- independent reference (flat tariffs from the config) --------------------


def _reference_payoffs(state, cfg, prices):
    a, b = cfg.export_price_inr_per_kwh, cfg.retail_price_inr_per_kwh
    f_s, f_b = cfg.seller_fee_inr_per_kwh, cfg.buyer_fee_inr_per_kwh
    remaining = {s: state.player(s).surplus_kwh for s in prices}
    sold = dict.fromkeys(prices, 0.0)
    for buyer in sorted(state.buyers):
        need = state.player(buyer).deficit_kwh
        acceptable = [s for s in prices if prices[s] + f_b < b - cfg.tol]
        for seller in sorted(acceptable, key=lambda s: (prices[s] + f_b, s)):
            q = min(remaining[seller], need)
            if q > cfg.tol:
                sold[seller] += q
                remaining[seller] -= q
                need -= q
    return {s: (prices[s] - a - f_s) * sold[s] for s in prices}


def _reference_equilibria(state, cfg, grid):
    sellers = state.sellers
    found = []
    for profile in itertools.product(grid, repeat=len(sellers)):
        prices = dict(zip(sellers, profile))
        current = _reference_payoffs(state, cfg, prices)
        stable = all(
            _reference_payoffs(state, cfg, {**prices, seller: alternative})[seller] <= current[seller] + cfg.tol
            for seller in sellers
            for alternative in grid
        )
        if stable:
            found.append(profile)
    return tuple(found)


def _assert_matches_reference(state, cfg, analysis):
    grid = analysis.price_grids[0]
    assert analysis.equilibria == _reference_equilibria(state, cfg, grid)
    for profile in itertools.product(grid, repeat=len(state.sellers)):
        expected = _reference_payoffs(state, cfg, dict(zip(state.sellers, profile)))
        assert analysis.payoffs(profile) == pytest.approx(expected, abs=1e-9)
    for equilibrium in analysis.equilibria:
        assert analysis.maximum_unilateral_regret(equilibrium) <= cfg.tol


# --- strategy sets -----------------------------------------------------------


def test_default_price_grid_matches_blueprint():
    assert price_grid(3.0, 8.0, 0.5) == DEFAULT_GRID


@pytest.mark.parametrize(
    "args,expected_first,expected_last,count",
    [
        ((3.0, 8.0, 0.5, 1.0, 1.0), 4.0, 6.5, 6),  # p - f_s >= 3 and p + f_b < 8
        ((3.0, 8.0, 0.25), 3.0, 7.75, 20),
        ((3.0, 8.0, 1.0), 3.0, 7.0, 5),
        ((3.3, 8.0, 0.5), 3.5, 7.5, 9),  # reservation price between ticks
    ],
)
def test_price_grid_respects_individual_rationality(args, expected_first, expected_last, count):
    grid = price_grid(*args)
    assert (grid[0], grid[-1], len(grid)) == (expected_first, expected_last, count)


def test_price_grid_edge_cases():
    assert price_grid(3.0, 8.0, 0.5, 2.5, 2.5) == ()  # fees absorb the whole spread
    assert price_grid(3.0, 8.0, 0.1)[:3] == (3.0, 3.1, 3.2)  # no float noise such as 3.1000000000000005
    with pytest.raises(ValueError, match="price step"):
        price_grid(3.0, 8.0, 0.0)


# --- buyer rule --------------------------------------------------------------


def test_buyers_take_cheapest_offers_first():
    trades = allocate_buyers(
        {"MG1": 6.0, "MG3": 5.0}, {"MG1": 20.0, "MG3": 10.0}, {"MG2": 15.0, "MG4": 25.0}, retail_price=8.0
    )
    assert trades == (("MG3", "MG2", 10.0), ("MG1", "MG2", 5.0), ("MG1", "MG4", 15.0))


def test_equal_prices_are_broken_by_seller_id():
    trades = allocate_buyers({"MG3": 6.0, "MG1": 6.0}, {"MG1": 5.0, "MG3": 5.0}, {"MG2": 4.0}, retail_price=8.0)
    assert trades == (("MG1", "MG2", 4.0),)


def test_offers_at_or_above_retail_are_never_accepted():
    supply, demand = {"MG1": 10.0}, {"MG2": 10.0}
    assert allocate_buyers({"MG1": 8.0}, supply, demand, retail_price=8.0) == ()
    assert allocate_buyers({"MG1": 7.5}, supply, demand, retail_price=8.0, buyer_fee=0.5) == ()
    assert allocate_buyers({"MG1": 7.49}, supply, demand, retail_price=8.0, buyer_fee=0.5) == (("MG1", "MG2", 10.0),)


def test_buyer_rule_respects_capacities():
    supply = {"MG1": 7.0, "MG3": 4.0}
    demand = {"MG2": 6.0, "MG4": 9.0}
    trades = allocate_buyers({"MG1": 5.0, "MG3": 4.5}, supply, demand, retail_price=8.0)
    for seller, cap in supply.items():
        assert sum(q for s, _, q in trades if s == seller) <= cap + 1e-12
    for buyer, cap in demand.items():
        assert sum(q for _, b, q in trades if b == buyer) <= cap + 1e-12
    assert sum(q for *_, q in trades) == pytest.approx(11.0)  # all supply sold; MG4 imports the rest


# --- blueprint acceptance checks ---------------------------------------------


def test_one_seller_selects_its_best_response():  # blueprint section 20 unit test
    cfg, state = _game([10, 0], [0, 10])
    analysis = analyze_price_game(state, cfg, price_grid_override=[3, 4, 5, 6, 7, 7.5])
    assert analysis.equilibria == ((7.5,),)
    assert analysis.payoffs((7.5,)) == {"MG1": pytest.approx(45.0)}  # (7.5 - 3) * 10
    assert analysis.examined_profiles == 6
    result = solve_competitive(state, cfg, price_grid_override=[3, 4, 5, 6, 7, 7.5])
    assert [(t.seller_id, t.buyer_id, t.quantity_kwh, t.price_inr_per_kwh) for t in result.trades] == [
        ("MG1", "MG2", 10.0, 7.5)
    ]
    assert result.settlement_for("MG1").gain_inr == pytest.approx(45.0)
    assert result.settlement_for("MG2").gain_inr == pytest.approx(5.0)


def test_changing_a_price_changes_sales_and_payoff():
    cfg, state = _game([20, 20, 0], [0, 0, 10])
    analysis = analyze_price_game(state, cfg)
    assert analysis.payoffs((4.0, 5.0)) == {"MG1": pytest.approx(10.0), "MG2": 0.0}
    assert analysis.payoffs((5.0, 4.0)) == {"MG1": 0.0, "MG2": pytest.approx(10.0)}


def test_seller_payoff_is_relative_to_export_and_seller_fee(golden_state, golden_cfg):
    cfg = dataclasses.replace(golden_cfg, seller_fee_inr_per_kwh=1.0, buyer_fee_inr_per_kwh=1.0)
    analysis = analyze_price_game(golden_state, cfg)
    assert analysis.equilibria == ((6.5, 6.5),)
    assert analysis.payoffs((6.5, 6.5))["MG1"] == pytest.approx((6.5 - 3.0 - 1.0) * 20)  # 50, not revenue 130


# --- T11: every reported equilibrium is verified -----------------------------


@pytest.mark.parametrize(
    "generation,demand",
    [
        ([40, 10, 25, 5], [20, 25, 15, 30]),  # golden: excess demand
        ([10, 10, 0], [0, 0, 12]),  # capacity-constrained price war
        ([20, 20, 0], [0, 0, 10]),  # Bertrand: each seller can serve all demand
        ([10, 10, 10, 0], [0, 0, 0, 25]),  # three sellers, 1000 profiles
        ([15, 0, 8, 0], [0, 6, 0, 14]),
    ],
)
def test_equilibria_match_independent_enumeration(generation, demand):
    cfg, state = _game(generation, demand)
    analysis = analyze_price_game(state, cfg)
    assert analysis.examined_profiles == len(DEFAULT_GRID) ** len(state.sellers)
    _assert_matches_reference(state, cfg, analysis)


def test_equilibria_match_reference_on_trading_hours(sample_profiles, default_cfg):
    for state in compute_hour_states(sample_profiles, default_cfg.tol):
        if state.sellers and state.buyers:
            _assert_matches_reference(state, default_cfg, analyze_price_game(state, default_cfg))


def test_settled_profile_is_a_verified_equilibrium(golden_state, golden_cfg):
    result = solve_competitive(golden_state, golden_cfg)
    analysis = analyze_price_game(golden_state, golden_cfg)
    selected = tuple(price for _, price in result.diagnostics.selected_profile)
    assert selected in analysis.equilibria
    assert result.diagnostics.pure_ne_found is True
    assert result.diagnostics.max_regret_inr <= golden_cfg.tol
    assert result.diagnostics.examined_profiles == 100


# --- T12: no pure equilibrium ------------------------------------------------


def test_price_war_has_no_pure_equilibrium_and_none_is_claimed():
    cfg, state = _game([10, 10, 0], [0, 0, 12])
    analysis = analyze_price_game(state, cfg)
    assert analysis.equilibria == () and analysis.pure_ne_found is False
    assert float(analysis.max_regret.min()) > cfg.tol  # every profile has a profitable deviation
    result = solve_competitive(state, cfg)
    d = result.diagnostics
    assert d.status == STATUS_NO_PURE_NE
    assert d.pure_ne_found is False and d.equilibrium_count == 0 and d.selected_profile is None
    assert result.trades == ()
    assert all(s.gain_inr == 0.0 for s in result.settlements)
    assert "NO PURE EQUILIBRIUM FOUND" in d.message and "NOT NASH" in d.message
    diagnostic = tuple(price for _, price in d.not_nash_profile)
    assert d.max_regret_inr == pytest.approx(analysis.maximum_unilateral_regret(diagnostic))
    assert d.max_regret_inr == pytest.approx(float(analysis.max_regret.min()))
    assert d.max_regret_inr > cfg.tol


# --- multiple equilibria and the tie rule ------------------------------------


def test_multiple_equilibria_are_counted_and_selected_deterministically():
    cfg, state = _game([20, 20, 0], [0, 0, 10])
    analysis = analyze_price_game(state, cfg)
    assert set(analysis.equilibria) == {(3.0, 3.0), (3.5, 3.0), (3.5, 3.5)}
    result = solve_competitive(state, cfg)
    assert result.diagnostics.equilibrium_count == 3
    assert result.diagnostics.selected_profile == (("MG1", 3.0), ("MG2", 3.0))  # lowest price sum
    assert [(t.seller_id, t.quantity_kwh, t.price_inr_per_kwh) for t in result.trades] == [("MG1", 10.0, 3.0)]


def test_tie_rule_decides_outcomes_between_identical_sellers():
    cfg, state = _game([20, 20, 0], [0, 0, 10])
    analysis = analyze_price_game(state, cfg)
    # identical sellers, yet only the lower ID wins ties: the equilibrium set is not symmetric
    assert (3.5, 3.0) in analysis.equilibria and (3.0, 3.5) not in analysis.equilibria
    assert analysis.payoffs((3.5, 3.5)) == {"MG1": pytest.approx(5.0), "MG2": 0.0}


# --- no game, enumeration cap ------------------------------------------------


@pytest.mark.parametrize("generation,demand", [([0, 0, 0], [5, 6, 7]), ([9, 9, 9], [1, 2, 3])])
def test_no_sellers_or_no_buyers(generation, demand):
    cfg, state = _game(generation, demand)
    result = solve_competitive(state, cfg)
    assert result.trades == () and result.diagnostics.status == "no_trade"
    assert result.diagnostics.pure_ne_found is None and result.diagnostics.examined_profiles == 0


def test_fees_that_absorb_the_spread_leave_no_game(golden_state, golden_cfg):
    cfg = dataclasses.replace(golden_cfg, seller_fee_inr_per_kwh=2.5, buyer_fee_inr_per_kwh=2.5)
    result = solve_competitive(golden_state, cfg)
    assert result.trades == () and result.diagnostics.status == "no_trade"
    assert "no individually rational price" in result.diagnostics.message
    assert result.total_cost_inr == pytest.approx(230.0)


def test_enumeration_cap_is_reported_not_guessed(golden_state, golden_cfg):
    cfg = dataclasses.replace(golden_cfg, max_competitive_sellers=1)
    assert analyze_price_game(golden_state, cfg).examined_profiles == 0
    with pytest.warns(CombinatorialWarning, match="exact enumeration not attempted"):
        result = solve_competitive(golden_state, cfg)
    d = result.diagnostics
    assert d.status == STATUS_CAP_EXCEEDED and d.pure_ne_found is None and d.examined_profiles == 0
    assert result.trades == ()


# --- 24-hour behaviour -------------------------------------------------------


@pytest.fixture
def competitive_results(sample_profiles, default_cfg):
    return run_competitive(sample_profiles, default_cfg)


def test_24h_equilibrium_status_regression(competitive_results):
    statuses = {r.hour: r.diagnostics.status for r in competitive_results}
    assert [h for h, s in statuses.items() if s == "success"] == [7, 8, 9, 10, 13, 14, 15, 16]
    assert [h for h, s in statuses.items() if s == STATUS_NO_PURE_NE] == [11, 12]
    for r in competitive_results:
        if r.diagnostics.status == "success":
            assert r.diagnostics.pure_ne_found is True and r.diagnostics.max_regret_inr <= 1e-8
        if r.diagnostics.status == STATUS_NO_PURE_NE:
            assert r.trades == () and r.diagnostics.not_nash_profile is not None


def test_lp_optimum_bounds_competitive_gain(competitive_results, sample_profiles, default_cfg):
    cooperative = run_cooperative(sample_profiles, default_cfg)
    for comp, coop in zip(competitive_results, cooperative):
        assert comp.total_gain_inr <= coop.total_gain_inr + 1e-7
        assert comp.total_p2p_volume_kwh <= coop.total_p2p_volume_kwh + 1e-7


def test_competitive_participation_is_voluntary(competitive_results):
    for r in competitive_results:
        assert all(s.gain_inr >= -1e-9 for s in r.settlements)
        for t in r.trades:
            assert t.seller_net_price_inr_per_kwh >= 3.0 and t.buyer_delivered_price_inr_per_kwh < 8.0


def test_deterministic_and_inputs_unchanged(sample_profiles, default_cfg):  # T13, T16
    snapshot = sample_profiles.copy(deep=True)
    assert run_competitive(sample_profiles, default_cfg) == run_competitive(sample_profiles, default_cfg)
    pd.testing.assert_frame_equal(sample_profiles, snapshot)


def test_six_microgrid_day_reports_every_interval_honestly():
    cfg = ScenarioConfig(microgrid_ids=tuple(f"MG{i}" for i in range(1, 7)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CombinatorialWarning)
        results = run_competitive(generate_profiles(cfg), cfg)
    for r in results:
        d = r.diagnostics
        assert d.status in {"success", "no_trade", STATUS_NO_PURE_NE, STATUS_CAP_EXCEEDED}
        if d.status == "success":
            assert d.pure_ne_found and d.max_regret_inr <= cfg.tol
            assert d.examined_profiles == len(DEFAULT_GRID) ** len(d.selected_profile)
        else:
            assert r.trades == ()
