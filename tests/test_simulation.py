"""Simulation controller, full metrics and diagnostics export (blueprint sections 5, 6, 9, 10; T16)."""

from __future__ import annotations

import dataclasses
import json

import pandas as pd
import pytest

import microgrid.simulation as simulation
from microgrid import cli
from microgrid.baseline import solve_baseline
from microgrid.cooperative import run_cooperative
from microgrid.data import diagnostics_records, validate_profiles
from microgrid.metrics import METRIC_COLUMNS, compute_metrics
from microgrid.models import DIAGNOSTICS_CSV_COLUMNS, ScenarioConfig
from microgrid.simulation import MECHANISMS, compare_all


def _with_fees(cfg, seller_fee, buyer_fee):
    return dataclasses.replace(cfg, seller_fee_inr_per_kwh=seller_fee, buyer_fee_inr_per_kwh=buyer_fee)


# --- controller --------------------------------------------------------------


def test_every_mechanism_runs_on_identical_states(sample_profiles, default_cfg):
    comparison = compare_all(sample_profiles, default_cfg)
    assert comparison.mechanisms == MECHANISMS
    intervals = {m: [(r.day, r.hour) for r in comparison.results_for(m)] for m in MECHANISMS}
    assert intervals["baseline"] == intervals["cooperative"] == intervals["competitive"] == [(0, h) for h in range(24)]
    baseline = comparison.results_for("baseline")
    for mechanism in MECHANISMS:
        for base, other in zip(baseline, comparison.results_for(mechanism)):
            assert [s.baseline_cost_inr for s in other.settlements] == [s.baseline_cost_inr for s in base.settlements]
    assert {name for name, _ in comparison.runtimes_seconds} == set(MECHANISMS)
    assert all(seconds >= 0 for _, seconds in comparison.runtimes_seconds)
    assert {"microgrid", "python", "platform", "numpy", "pandas", "scipy"} <= set(dict(comparison.environment))


def test_input_profiles_are_unchanged(sample_profiles, default_cfg):  # T16
    snapshot = sample_profiles.copy(deep=True)
    compare_all(sample_profiles, default_cfg)
    pd.testing.assert_frame_equal(sample_profiles, snapshot)


def test_a_mechanism_that_mutates_the_input_is_caught(monkeypatch, golden_profiles, golden_cfg):
    frame = golden_profiles.copy()

    def mutating(state, cfg, tariffs):
        frame.loc[0, "demand_kwh"] = 999.0
        return solve_baseline(state, cfg, tariffs)

    monkeypatch.setitem(simulation._SOLVERS, "baseline", mutating)
    with pytest.raises(RuntimeError, match="modified the input profiles"):
        compare_all(frame, golden_cfg)


def test_mechanism_selection(golden_profiles, golden_cfg):
    with pytest.raises(ValueError, match="unknown mechanism"):
        compare_all(golden_profiles, golden_cfg, mechanisms=("auction",))
    only = compare_all(golden_profiles, golden_cfg, mechanisms=("competitive",))
    assert only.mechanisms == ("competitive",)
    assert only.metrics["competitive"]["efficiency"] is None  # no LP reference without the cooperative run
    with pytest.raises(KeyError):
        only.results_for("cooperative")


def test_controller_matches_the_individual_mechanisms(sample_profiles, default_cfg):
    comparison = compare_all(sample_profiles, default_cfg)
    assert comparison.results_for("cooperative") == tuple(run_cooperative(sample_profiles, default_cfg))


# --- metrics -----------------------------------------------------------------


def test_golden_metrics(golden_profiles, golden_cfg):
    m = compare_all(golden_profiles, golden_cfg).metrics
    assert [m[k]["net_community_cost_inr"] for k in MECHANISMS] == pytest.approx([230.0, 80.0, 80.0])
    assert [m[k]["total_gain_inr"] for k in MECHANISMS] == pytest.approx([0.0, 150.0, 150.0])
    assert m["cooperative"]["cost_reduction_pct"] == pytest.approx(100 * 150 / 230)
    assert m["cooperative"]["average_price_inr_per_kwh"] == pytest.approx(5.5)
    assert m["competitive"]["average_price_inr_per_kwh"] == pytest.approx(7.5)
    assert m["cooperative"]["per_player_gain_inr"] == pytest.approx({"MG1": 50.0, "MG2": 37.5, "MG3": 25.0, "MG4": 37.5})
    assert m["competitive"]["per_player_gain_inr"] == pytest.approx({"MG1": 90.0, "MG2": 7.5, "MG3": 45.0, "MG4": 7.5})
    assert all(m[k]["lp_optimal_gain_inr"] == pytest.approx(150.0) for k in MECHANISMS)
    assert (m["baseline"]["efficiency"], m["cooperative"]["efficiency"], m["competitive"]["efficiency"]) == (
        0.0, pytest.approx(1.0), pytest.approx(1.0))
    assert m["cooperative"]["jain_fairness"] == pytest.approx(0.94736842)
    assert m["competitive"]["pure_ne_intervals"] == 1 and m["competitive"]["max_regret_inr"] == pytest.approx(0.0)
    assert m["baseline"]["pure_ne_intervals"] is None and m["cooperative"]["max_regret_inr"] is None


def test_zero_denominators_are_not_applicable(golden_profiles, golden_cfg):
    m = compare_all(golden_profiles, _with_fees(golden_cfg, 2.5, 2.5)).metrics
    for name in MECHANISMS:
        assert m[name]["total_p2p_volume_kwh"] == 0.0
        assert m[name]["average_price_inr_per_kwh"] is None  # no 0/0 price
        assert m[name]["jain_fairness"] is None  # all gains zero
        assert m[name]["efficiency"] is None  # LP optimum is zero
        assert m[name]["cost_reduction_pct"] == 0.0  # baseline is positive, so a percentage is defined


def test_cost_reduction_is_not_applicable_when_baseline_is_not_positive():
    cfg = ScenarioConfig(scenario_id="exporters", microgrid_ids=("MG1", "MG2"), require_full_horizon=False)
    frame = pd.DataFrame({"day": 0, "hour": 12, "microgrid_id": ["MG1", "MG2"],
                          "generation_kwh": [10.0, 8.0], "demand_kwh": [2.0, 1.0]})
    m = compare_all(validate_profiles(frame, cfg), cfg).metrics
    assert m["baseline"]["baseline_cost_inr"] < 0  # net export income
    assert all(m[name]["cost_reduction_pct"] is None for name in MECHANISMS)


def test_compute_metrics_rejects_empty_or_mixed_input(golden_profiles, golden_cfg):
    comparison = compare_all(golden_profiles, golden_cfg)
    with pytest.raises(ValueError, match="at least one"):
        compute_metrics([])
    mixed = comparison.results_for("baseline") + comparison.results_for("cooperative")
    with pytest.raises(ValueError, match="one mechanism"):
        compute_metrics(mixed)


def test_competitive_equilibrium_counts_on_24h(sample_profiles, default_cfg):
    m = compare_all(sample_profiles, default_cfg).metrics
    comp = m["competitive"]
    assert (comp["pure_ne_intervals"], comp["no_pure_ne_intervals"], comp["cap_exceeded_intervals"]) == (8, 2, 0)
    assert comp["max_regret_inr"] <= default_cfg.tol
    assert comp["max_not_nash_regret_inr"] == pytest.approx(12.7588, abs=1e-4)  # worst of hours 11 and 12
    assert comp["efficiency"] == pytest.approx(comp["total_gain_inr"] / m["cooperative"]["total_gain_inr"])


# --- tables and diagnostics ----------------------------------------------------


def test_output_tables(sample_profiles, default_cfg):
    comparison = compare_all(sample_profiles, default_cfg)
    summary = comparison.summary_frame()
    assert tuple(summary.columns) == METRIC_COLUMNS and summary.mechanism.tolist() == list(MECHANISMS)
    players = comparison.player_frame()
    assert players.columns[0] == "mechanism" and len(players) == 3 * 4
    assert set(comparison.trades_frame().mechanism) == {"cooperative", "competitive"}
    assert len(comparison.settlement_frame()) == 3 * 96
    diagnostics = comparison.diagnostics_frame()
    assert tuple(diagnostics.columns) == DIAGNOSTICS_CSV_COLUMNS and len(diagnostics) == 3 * 24
    competitive = diagnostics[diagnostics.mechanism == "competitive"].set_index("hour")
    assert json.loads(competitive.loc[9, "selected_profile"]) == {"MG1": 7.5, "MG3": 7.5}
    assert json.loads(competitive.loc[12, "not_nash_profile"]) == {"MG1": 7.0, "MG3": 6.0}
    assert competitive.loc[12, "selected_profile"] == ""


def test_diagnostics_records_follow_the_blueprint_format(golden_profiles, golden_cfg):
    comparison = compare_all(golden_profiles, golden_cfg)
    record = diagnostics_records(comparison.results_for("competitive"))[0]
    expected = {
        "mechanism": "competitive",
        "day": 0,
        "hour": 12,
        "pure_ne_found": True,
        "equilibrium_count": 1,
        "selected_profile": {"MG1": 7.5, "MG3": 7.5},
        "max_unilateral_regret_inr": 0.0,
        "examined_profiles": 100,
        "status": "success",
    }
    assert {k: record[k] for k in expected} == expected
    json.dumps(diagnostics_records(r for _, rs in comparison.results for r in rs))  # JSON-serialisable


# --- CLI ---------------------------------------------------------------------


def test_cli_compare_writes_every_table(repo_root, tmp_path, capsys):
    code = cli.main(
        [
            "compare",
            "--config", str(repo_root / "configs" / "golden_one_hour.json"),
            "--profiles", str(repo_root / "data" / "golden_one_hour.csv"),
            "--out-dir", str(tmp_path),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    cost_row = next(line for line in out.splitlines() if line.startswith("Community net cost (INR)"))
    assert cost_row.split()[-3:] == ["230.00", "80.00", "80.00"]
    assert "Efficiency vs LP optimum" in out and "N/A" in out
    stem = tmp_path / "golden_one_hour_compare"
    for suffix in ("summary.csv", "players.csv", "trades.csv", "settlement.csv", "diagnostics.csv",
                   "diagnostics.json", "run_info.json"):
        assert (tmp_path / f"golden_one_hour_compare_{suffix}").is_file(), suffix
    assert tuple(pd.read_csv(f"{stem}_summary.csv").columns) == METRIC_COLUMNS
    info = json.loads((tmp_path / "golden_one_hour_compare_run_info.json").read_text())
    assert set(info) == {"environment", "runtimes_seconds", "mechanisms"}
    assert (tmp_path / "golden_one_hour_config.json").is_file()
