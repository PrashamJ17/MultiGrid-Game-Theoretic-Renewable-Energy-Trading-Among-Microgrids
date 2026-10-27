"""Surplus/deficit correctness and energy-conservation properties (blueprint T06, T10)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microgrid.baseline import run_baseline
from microgrid.data import results_to_settlement_frame
from microgrid.energy import (
    add_surplus_deficit_columns,
    check_player_balance,
    clamp_tol,
    classify,
    deficit,
    surplus,
)
from microgrid.models import EnergyBalanceError, Role

KEYS = ["day", "hour", "microgrid_id"]


# --- surplus and deficit ----------------------------------------------------


@pytest.mark.parametrize(
    "g,d,s,r",
    [
        (40, 20, 20, 0),
        (10, 25, 0, 15),
        (25, 15, 10, 0),
        (5, 30, 0, 25),
        (7, 7, 0, 0),
        (0, 0, 0, 0),
        (0, 12.5, 0, 12.5),
        (3.25, 0, 3.25, 0),
    ],
)
def test_surplus_and_deficit_scalars(g, d, s, r):
    assert surplus(g, d) == s
    assert deficit(g, d) == r
    assert isinstance(surplus(g, d), float) and isinstance(deficit(g, d), float)


def test_surplus_and_deficit_are_vectorised():
    g = np.array([40.0, 10.0, 25.0, 5.0])
    d = np.array([20.0, 25.0, 15.0, 30.0])
    np.testing.assert_array_equal(surplus(g, d), [20.0, 0.0, 10.0, 0.0])
    np.testing.assert_array_equal(deficit(g, d), [0.0, 15.0, 0.0, 25.0])
    gs = pd.Series(g, index=list("abcd"))
    ds = pd.Series(d, index=list("abcd"))
    result = surplus(gs, ds)
    assert isinstance(result, pd.Series) and list(result.index) == list("abcd")


def test_golden_totals_and_roles(golden_state):
    assert golden_state.total_generation_kwh == 80.0
    assert golden_state.total_demand_kwh == 90.0
    assert golden_state.total_surplus_kwh == 30.0
    assert golden_state.total_deficit_kwh == 40.0
    assert golden_state.sellers == ("MG1", "MG3")
    assert golden_state.buyers == ("MG2", "MG4")


def test_identities_on_random_inputs():
    rng = np.random.default_rng(2024)
    g = rng.uniform(0.0, 50.0, 10_000)
    d = rng.uniform(0.0, 50.0, 10_000)
    g[:100] = d[:100]  # exact ties
    s, r = surplus(g, d), deficit(g, d)
    assert (s >= 0).all() and (r >= 0).all()
    assert not ((s > 0) & (r > 0)).any(), "a microgrid can never be seller and buyer at once"
    np.testing.assert_allclose(s - r, g - d, rtol=0, atol=1e-12)
    assert (s[:100] == 0).all() and (r[:100] == 0).all()


def test_sample_profile_identities(sample_profiles, default_cfg):
    e = add_surplus_deficit_columns(sample_profiles, tol=default_cfg.tol)
    np.testing.assert_allclose(e.surplus_kwh - e.deficit_kwh, e.generation_kwh - e.demand_kwh, atol=1e-12)
    assert not ((e.surplus_kwh > 0) & (e.deficit_kwh > 0)).any()
    assert set(e.loc[e.surplus_kwh > default_cfg.tol, "role"]) <= {Role.SELLER.value}
    assert set(e.loc[e.deficit_kwh > default_cfg.tol, "role"]) <= {Role.BUYER.value}


# --- conservation under the baseline ----------------------------------------


def _baseline_flows(profiles, cfg) -> pd.DataFrame:
    settlement = results_to_settlement_frame(run_baseline(profiles, cfg))
    return profiles.merge(settlement, on=KEYS, validate="one_to_one")


def test_player_energy_identity_every_hour(sample_profiles, default_cfg):
    flows = _baseline_flows(sample_profiles, default_cfg)
    assert len(flows) == 96
    supply = flows.generation_kwh + flows.grid_import_kwh + flows.p2p_import_kwh
    use = flows.demand_kwh + flows.grid_export_kwh + flows.p2p_export_kwh
    np.testing.assert_allclose(supply, use, rtol=0, atol=1e-9)
    assert not ((flows.grid_import_kwh > 0) & (flows.grid_export_kwh > 0)).any()
    assert (flows[["grid_import_kwh", "grid_export_kwh"]] >= 0).all().all()


def test_community_energy_identity_every_hour(sample_profiles, default_cfg):
    hourly = _baseline_flows(sample_profiles, default_cfg).groupby(["day", "hour"]).sum(numeric_only=True)
    np.testing.assert_allclose(
        hourly.generation_kwh + hourly.grid_import_kwh,
        hourly.demand_kwh + hourly.grid_export_kwh,
        rtol=0,
        atol=1e-9,
    )


def test_golden_conservation(golden_profiles, golden_cfg):
    flows = _baseline_flows(golden_profiles, golden_cfg)
    assert flows.generation_kwh.sum() + flows.grid_import_kwh.sum() == 80.0 + 40.0
    assert flows.demand_kwh.sum() + flows.grid_export_kwh.sum() == 90.0 + 30.0


# --- balance checker, tolerance helpers --------------------------------------


def test_balance_check_accepts_consistent_flows():
    check_player_balance(
        microgrid_id="MG1", generation_kwh=40, demand_kwh=20, grid_import_kwh=0, grid_export_kwh=20
    )
    check_player_balance(  # float drift far below tolerance is accepted
        microgrid_id="MG1", generation_kwh=40, demand_kwh=20, grid_import_kwh=0, grid_export_kwh=20 + 1e-12
    )


@pytest.mark.parametrize(
    "flows,match",
    [
        (dict(generation_kwh=40, demand_kwh=20, grid_import_kwh=0, grid_export_kwh=19), "balance violated"),
        (dict(generation_kwh=10, demand_kwh=25, grid_import_kwh=-1, grid_export_kwh=0), "negative"),
        (dict(generation_kwh=10, demand_kwh=25, grid_import_kwh=20, grid_export_kwh=5), "same interval"),
        (
            dict(generation_kwh=10, demand_kwh=10, grid_import_kwh=0, grid_export_kwh=0,
                 p2p_import_kwh=5, p2p_export_kwh=5),
            "P2P",
        ),
        (dict(generation_kwh=float("nan"), demand_kwh=10, grid_import_kwh=0, grid_export_kwh=0), "not finite"),
    ],
)
def test_balance_check_rejects_inconsistent_flows(flows, match):
    with pytest.raises(EnergyBalanceError, match=match):
        check_player_balance(microgrid_id="MGx", **flows)


def test_clamp_tol():
    assert clamp_tol(5.0) == 5.0
    assert clamp_tol(0.0) == 0.0
    assert clamp_tol(-1e-10, tol=1e-8) == 0.0
    with pytest.raises(EnergyBalanceError, match="negative beyond tolerance"):
        clamp_tol(-1e-6, tol=1e-8)
    with pytest.raises(EnergyBalanceError, match="not finite"):
        clamp_tol(float("nan"))


def test_classify_uses_tolerance():
    assert classify(5e-9, 0.0, tol=1e-8) is Role.BALANCED
    assert classify(2e-8, 0.0, tol=1e-8) is Role.SELLER
    assert classify(0.0, 1.0) is Role.BUYER
    assert classify(0.0, 0.0) is Role.BALANCED
    with pytest.raises(EnergyBalanceError):
        classify(1.0, 1.0)
