"""Smoke tests: package import, CLI, Streamlit boot and architecture rules."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys

import pandas as pd
import pytest

import microgrid
from microgrid import cli, competitive, cooperative, metrics, settlement, simulation
from microgrid.data import load_config
from microgrid.models import SETTLEMENT_CSV_COLUMNS, ScenarioConfig

UI_LIBRARIES = {"streamlit", "plotly", "matplotlib"}


def test_package_imports():
    assert microgrid.__version__ == "0.4.0"
    for name in microgrid.__all__:
        assert hasattr(microgrid, name), name
    assert callable(cooperative.solve_cooperative) and callable(settlement.settle_market)
    assert callable(competitive.solve_competitive)


def test_controller_and_metrics_on_golden_fixture(golden_cfg, golden_profiles):
    comparison = simulation.compare_all(golden_profiles, golden_cfg)
    assert comparison.mechanisms == ("baseline", "cooperative", "competitive")
    m = comparison.metrics
    assert [m[k]["net_community_cost_inr"] for k in comparison.mechanisms] == pytest.approx([230.0, 80.0, 80.0])
    assert m["cooperative"]["efficiency"] == pytest.approx(1.0) and m["baseline"]["efficiency"] == 0.0
    assert m["cooperative"]["cost_reduction_pct"] == pytest.approx(65.2173913)
    assert m["cooperative"]["average_price_inr_per_kwh"] == pytest.approx(5.5)
    assert m["baseline"]["average_price_inr_per_kwh"] is None  # no volume: N/A, not 0/0
    assert m["competitive"]["pure_ne_intervals"] == 1 and m["baseline"]["pure_ne_intervals"] is None
    assert m["competitive"]["jain_fairness"] == pytest.approx(22500 / 40950)
    diagnostics = comparison.diagnostics_frame()
    assert diagnostics.loc[diagnostics.mechanism == "competitive", "selected_profile"].item() == '{"MG1": 7.5, "MG3": 7.5}'
    with pytest.raises(ValueError):
        metrics.compute_metrics([])


def test_cli_baseline_on_golden_fixture(repo_root, tmp_path, capsys):
    config = repo_root / "configs" / "golden_one_hour.json"
    code = cli.main(
        [
            "baseline",
            "--config", str(config),
            "--profiles", str(repo_root / "data" / "golden_one_hour.csv"),
            "--out-dir", str(tmp_path),
        ]
    )
    assert code == 0
    assert "Community baseline net cost: INR 230.00" in capsys.readouterr().out
    table = pd.read_csv(tmp_path / "golden_one_hour_baseline_settlement.csv")
    assert tuple(table.columns) == SETTLEMENT_CSV_COLUMNS
    assert table.baseline_cost_inr.sum() == pytest.approx(230.0)
    assert pd.read_csv(tmp_path / "golden_one_hour_baseline_trades.csv").empty
    saved = json.loads((tmp_path / "golden_one_hour_config.json").read_text())
    assert ScenarioConfig.from_dict(saved) == load_config(config)


def test_cli_cooperative_on_golden_fixture(repo_root, tmp_path, capsys):
    code = cli.main(
        [
            "cooperative",
            "--config", str(repo_root / "configs" / "golden_one_hour.json"),
            "--profiles", str(repo_root / "data" / "golden_one_hour.csv"),
            "--out-dir", str(tmp_path),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "Community baseline net cost: INR 230.00" in out
    assert "Community cooperative net cost: INR 80.00" in out
    assert "Community gain: INR 150.00 (P2P volume 30.000 kWh; operator fees INR 0.00)" in out
    assert "Jain fairness of per-player gains: 0.94736842" in out
    trades = pd.read_csv(tmp_path / "golden_one_hour_cooperative_trades.csv")
    assert list(zip(trades.seller_id, trades.buyer_id, trades.quantity_kwh, trades.price_inr_per_kwh)) == [
        ("MG1", "MG2", 15.0, 5.5), ("MG1", "MG4", 5.0, 5.5), ("MG3", "MG4", 10.0, 5.5)
    ]
    table = pd.read_csv(tmp_path / "golden_one_hour_cooperative_settlement.csv")
    assert table.p2p_cost_inr.sum() == pytest.approx(80.0)


def test_cli_competitive_on_golden_fixture(repo_root, tmp_path, capsys):
    code = cli.main(
        [
            "competitive",
            "--config", str(repo_root / "configs" / "golden_one_hour.json"),
            "--profiles", str(repo_root / "data" / "golden_one_hour.csv"),
            "--out-dir", str(tmp_path),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "success" in out and "MG1=7.50, MG3=7.50" in out
    assert "Community competitive net cost: INR 80.00" in out
    assert "Jain fairness of per-player gains: 0.54945055" in out
    trades = pd.read_csv(tmp_path / "golden_one_hour_competitive_trades.csv")
    assert set(trades.price_inr_per_kwh) == {7.5} and trades.quantity_kwh.sum() == pytest.approx(30.0)


def test_cli_competitive_reports_missing_equilibria(repo_root, capsys):
    code = cli.main(
        [
            "competitive",
            "--config", str(repo_root / "configs" / "default.json"),
            "--profiles", str(repo_root / "data" / "sample_24h.csv"),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "NO_PURE_NE" in out and "NOT NASH (diagnostic)" in out
    assert "2 interval(s) have NO PURE EQUILIBRIUM" in out


def test_cli_generate_is_deterministic(repo_root, tmp_path):
    config = str(repo_root / "configs" / "default.json")
    assert cli.main(["generate", "--config", config, "--out", str(tmp_path / "a.csv")]) == 0
    assert cli.main(["generate", "--config", config, "--out", str(tmp_path / "b.csv")]) == 0
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()
    assert cli.main(["generate", "--config", config, "--seed", "7", "--out", str(tmp_path / "c.csv")]) == 0
    assert (tmp_path / "c.csv").read_bytes() != (tmp_path / "a.csv").read_bytes()


def test_cli_reports_invalid_input_without_traceback(repo_root, tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    bad.write_text("day,hour,microgrid_id,generation_kwh,demand_kwh\n0,12,MG1,-40,20\n")
    code = cli.main(
        ["baseline", "--config", str(repo_root / "configs" / "golden_one_hour.json"), "--profiles", str(bad)]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "negative energy" in err and "CSV line 2" in err


def test_cli_runs_as_module(repo_root):
    env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}
    proc = subprocess.run(
        [
            sys.executable, "-m", "microgrid.cli", "baseline",
            "--config", "configs/golden_one_hour.json",
            "--profiles", "data/golden_one_hour.csv",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "INR 230.00" in proc.stdout


def test_streamlit_app_boots(repo_root):
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_file(str(repo_root / "app.py"), default_timeout=60).run()
    assert not app.exception, app.exception
    assert "230.00" in [m.value for m in app.metric]  # golden baseline; behaviour is in test_dashboard.py


def _top_level_imports(path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_package_has_no_ui_dependencies(repo_root):
    for path in sorted((repo_root / "src" / "microgrid").glob("*.py")):
        leaked = _top_level_imports(path) & UI_LIBRARIES
        assert not leaked, f"{path.name} imports UI libraries {leaked}; only app.py may"


def test_models_layer_is_dependency_free(repo_root):
    path = repo_root / "src" / "microgrid" / "models.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    relative = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level > 0]
    assert not relative, "models.py must not import other microgrid modules"
    forbidden = {"pandas", "numpy", "microgrid"} | UI_LIBRARIES
    assert not (_top_level_imports(path) & forbidden)
