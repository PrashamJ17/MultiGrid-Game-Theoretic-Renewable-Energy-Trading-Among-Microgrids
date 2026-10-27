"""Experiments E1-E7, report figures and the independent export audit (blueprint sections 13 and 19, Prompt 6)."""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import sys

import pandas as pd
import pytest

from microgrid import cli
from microgrid.experiments import SPECS, profiles_sha256, run_experiment, save_experiment
from conftest import REPO_ROOT

FROZEN = ("E1", "E2", "E3", "E6")
VARIED_FIELDS = {
    "E1": set(),
    "E2": {"retail_price_inr_per_kwh"},
    "E3": {"seller_fee_inr_per_kwh", "buyer_fee_inr_per_kwh"},
    "E4": {"generator"},
    "E5": {"microgrid_ids", "generator"},
    "E6": {"price_step_inr"},
    "E7": {"seed"},
}


@pytest.fixture(scope="module")
def results():
    from microgrid.data import load_and_validate_profiles, load_config

    cfg = load_config(REPO_ROOT / "configs" / "default.json")
    profiles = load_and_validate_profiles(REPO_ROOT / "data" / "sample_24h.csv", cfg)
    return cfg, profiles, {e: run_experiment(e, cfg, profiles) for e in sorted(SPECS)}


@pytest.fixture(scope="module")
def saved(results, tmp_path_factory):
    root = tmp_path_factory.mktemp("experiments")
    for experiment_id, result in results[2].items():
        save_experiment(result, root / experiment_id)
    return root


# --- design rules -------------------------------------------------------------


def test_specs_match_the_blueprint_matrix():
    assert sorted(SPECS) == ["E1", "E2", "E3", "E4", "E5", "E6", "E7"]
    assert SPECS["E3"].values[0] == 0.0 and SPECS["E3"].values[-1] >= 5.0  # fees 0 to 5 INR/kWh
    assert SPECS["E4"].values == (0.5, 1.0, 1.5, 2.0)
    assert SPECS["E5"].values == (2, 4, 6)
    assert SPECS["E6"].values == (1.0, 0.5, 0.25)
    assert SPECS["E7"].values == tuple(range(42, 52))


@pytest.mark.parametrize("experiment_id", sorted(VARIED_FIELDS))
def test_each_experiment_varies_exactly_one_parameter_family(results, experiment_id):
    base = results[0].to_dict()
    for run in results[2][experiment_id].runs:
        changed = {k for k, v in run.config.to_dict().items() if v != base[k]} - {"scenario_id"}
        assert changed <= VARIED_FIELDS[experiment_id], (run.run_id, changed)


def test_profiles_are_frozen_unless_the_generator_is_varied(results):
    _, profiles, by_id = results
    reference = profiles_sha256(profiles)
    for experiment_id in FROZEN:
        assert {run.profiles_sha256 for run in by_id[experiment_id].runs} == {reference}, experiment_id
    for experiment_id in ("E4", "E5", "E7"):
        hashes = [run.profiles_sha256 for run in by_id[experiment_id].runs]
        assert len(set(hashes)) == len(hashes), experiment_id
    seed42 = next(run for run in by_id["E7"].runs if run.value == 42)
    assert seed42.profiles_sha256 == reference  # regenerating seed 42 reproduces the committed day


# --- results that follow from the model ------------------------------------------


def test_fee_sweep_is_linear_for_the_cooperative_market_and_stops_at_the_spread(results):
    runs = results[2]["E3"].runs_frame()
    coop = runs[runs.mechanism == "cooperative"].set_index("total_fee_inr_per_kwh")
    volume = coop.loc[0.0, "total_p2p_volume_kwh"]
    for fee, row in coop.iterrows():
        expected = max(5.0 - fee, 0.0) * volume if fee < 5.0 else 0.0
        assert row.total_gain_inr == pytest.approx(expected, abs=1e-6), fee
    beyond = runs[runs.total_fee_inr_per_kwh >= 5.0]
    assert (beyond.total_p2p_volume_kwh == 0).all() and (beyond.total_gain_inr == 0).all()


def test_spread_sweep_scales_cooperative_gain_with_the_spread(results):
    runs = results[2]["E2"].runs_frame()
    coop = runs[runs.mechanism == "cooperative"].set_index("spread_inr_per_kwh")
    volume = coop.loc[1.0, "total_p2p_volume_kwh"]
    for spread, row in coop.iterrows():
        assert row.total_gain_inr == pytest.approx(spread * volume if spread > 0 else 0.0, abs=1e-6)


def test_price_tick_changes_prices_but_not_volume_or_total_gain(results):
    runs = results[2]["E6"].runs_frame()
    comp = runs[runs.mechanism == "competitive"]
    assert comp.total_gain_inr.nunique() == 1 and comp.total_p2p_volume_kwh.nunique() == 1
    assert comp.average_price_inr_per_kwh.tolist() == pytest.approx([7.0, 7.5, 7.75])  # top tick below retail


def test_competitive_gain_never_exceeds_the_lp_optimum(results):
    for result in results[2].values():
        runs = result.runs_frame()
        for _, group in runs.groupby("run_id"):
            row = group.set_index("mechanism")
            assert row.loc["competitive", "total_gain_inr"] <= row.loc["cooperative", "total_gain_inr"] + 1e-6


def test_seed_summary_reports_every_seed(results):
    summary = results[2]["E7"].summary_frame()
    gain = summary[(summary.mechanism == "cooperative") & (summary.metric == "total_gain_inr")].iloc[0]
    assert gain.n == 10 and gain["min"] <= gain["mean"] <= gain["max"] and gain["std"] > 0


# --- saved outputs ----------------------------------------------------------------


def test_saved_tables_and_configs(saved):
    e1 = saved / "E1"
    for name in ("runs.csv", "players.csv", "intervals.csv", "summary.csv", "runtimes.csv", "experiment.json",
                 "profiles.csv", "trades.csv", "settlement.csv", "diagnostics.csv", "payoff_curves.csv"):
        assert (e1 / name).is_file(), name
    manifest = json.loads((saved / "E3" / "experiment.json").read_text())
    assert manifest["variable"] == "total_fee_inr_per_kwh" and len(manifest["runs"]) == len(SPECS["E3"].values)
    for run in manifest["runs"]:
        assert (saved / "E3" / run["config"]).is_file()
    assert "runtime_seconds" not in pd.read_csv(saved / "E3" / "runs.csv").columns  # timing kept separate


def test_result_tables_are_byte_identical_on_rerun(results, tmp_path):
    cfg, profiles, _ = results
    first = save_experiment(run_experiment("E3", cfg, profiles), tmp_path / "a")
    save_experiment(run_experiment("E3", cfg, profiles), tmp_path / "b")
    for path in first:
        if path.name in ("runtimes.csv", "experiment.json"):
            continue
        twin = tmp_path / "b" / path.relative_to(tmp_path / "a")
        assert path.read_bytes() == twin.read_bytes(), path.name


def test_cli_experiment(tmp_path, capsys):
    code = cli.main(["experiment", "E6", "--config", str(REPO_ROOT / "configs" / "default.json"),
                     "--profiles", str(REPO_ROOT / "data" / "sample_24h.csv"), "--out-dir", str(tmp_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "E6 Strategy resolution" in out and (tmp_path / "E6" / "runs.csv").is_file()
    assert cli.main(["experiment", "E9", "--out-dir", str(tmp_path)]) == 2


# --- figures and the independent audit ---------------------------------------------


def _script(name: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", str(REPO_ROOT / "scripts" / name), *args],
                          capture_output=True, text=True, timeout=300)


def test_report_figures_are_drawn_from_saved_results(saved, tmp_path):
    proc = _script("make_figures.py", "--results", str(saved), "--out", str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    pngs = sorted(tmp_path.glob("fig*.png"))
    assert len(pngs) == 12 and all(p.stat().st_size > 10_000 for p in pngs)


def test_export_audit_passes_on_genuine_results(saved):
    proc = _script("audit_exports.py", str(saved / "E1"))
    assert proc.returncode == 0, proc.stdout
    assert "22 passed, 0 failed" in proc.stdout


@pytest.mark.parametrize(
    "corrupt,expected",
    [
        (lambda d: d["trades"].assign(price_inr_per_kwh=d["trades"].price_inr_per_kwh + 0.25), "priced at the selected"),
        (lambda d: d["diagnostics"].assign(status=d["diagnostics"].status.replace("NO_PURE_NE", "success")),
         "status success"),
        (lambda d: d["settlement"].assign(p2p_cost_inr=d["settlement"].p2p_cost_inr - 1.0), "gain = baseline cost"),
    ],
)
def test_export_audit_detects_tampering(saved, tmp_path, corrupt, expected):
    folder = tmp_path / "E1"
    shutil.copytree(saved / "E1", folder)
    files = {"trades": "trades.csv", "diagnostics": "diagnostics.csv", "settlement": "settlement.csv"}
    frames = {k: pd.read_csv(folder / v) for k, v in files.items()}
    changed = corrupt(frames)
    target = next(k for k, frame in frames.items() if list(frame.columns) == list(changed.columns)
                  and not frame.equals(changed))
    changed.to_csv(folder / files[target], index=False)
    proc = _script("audit_exports.py", str(folder))
    assert proc.returncode == 1
    assert any(line.startswith("[FAIL]") and expected in line for line in proc.stdout.splitlines()), proc.stdout
