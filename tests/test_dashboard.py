"""Streamlit dashboard: boots, drives its controls, and shows the package's results.

The app is a thin adapter, so these tests check wiring and honesty rules:
the right numbers appear, controls re-run the mechanisms, profiles stay
fixed when only tariffs change, and missing equilibria are flagged. The
numbers themselves are tested elsewhere.
"""

from __future__ import annotations

import pandas as pd
import pytest

testing = pytest.importorskip("streamlit.testing.v1")

GOLDEN = "Golden one-hour fixture (blueprint section 4)"
SAMPLE = "Synthetic 24 h, seed 42 (committed CSV)"
CUSTOM = "Synthetic 24 h, custom seed and solar scale"
MECHANISM_LABELS = (
    "Baseline (utility only)",
    "Cooperative (LP + Nash bargaining)",
    "Competitive (seller price game)",
)


@pytest.fixture
def app(repo_root):
    app = testing.AppTest.from_file(str(repo_root / "app.py"), default_timeout=120).run()
    assert not app.exception, app.exception
    return app


def _metric(app, label: str) -> str:
    return next(m.value for m in app.metric if m.label == label)


def _costs(app) -> list[str]:
    return [_metric(app, label) for label in MECHANISM_LABELS]


def _table_with(app, column: str) -> pd.DataFrame:
    return next(d.value for d in app.dataframe if column in d.value.columns)


def test_default_view_is_the_golden_fixture(app):
    assert [t.label for t in app.tabs] == [
        "Overview", "Profiles", "Simulate", "Transactions", "Game analysis", "Comparison", "Network", "Downloads",
    ]
    assert _costs(app) == ["230.00", "80.00", "80.00"]
    assert _metric(app, "Microgrids") == "4"
    assert any("Verified pure Nash equilibrium: MG1=7.50, MG3=7.50" in s.value for s in app.success)
    assert any(w.value.startswith("Simulation only.") for w in app.warning)


def test_fees_that_absorb_the_spread_stop_all_trading(app):
    app.sidebar.slider(key="seller_fee").set_value(2.5).run()
    app.sidebar.slider(key="buyer_fee").set_value(2.5).run()
    assert not app.exception
    assert _costs(app) == ["230.00", "230.00", "230.00"]
    assert any("No P2P trades under the cooperative mechanism" in i.value for i in app.info)


def test_alpha_moves_the_cooperative_price_but_not_the_total(app):
    app.sidebar.slider(key="alpha").set_value(0.8).run()
    assert not app.exception
    assert _costs(app)[1] == "80.00"
    trades = _table_with(app, "price_inr_per_kwh")
    assert not trades.empty
    assert all(price == pytest.approx(7.0) for price in trades.price_inr_per_kwh)  # 3 + 0.8 * 5


def test_tariff_changes_do_not_regenerate_profiles(app):
    app.sidebar.selectbox(key="scenario").set_value(SAMPLE).run()
    before = _table_with(app, "generation_kwh")[["day", "hour", "microgrid_id", "generation_kwh", "demand_kwh"]]
    app.sidebar.slider(key="retail").set_value(10.0).run()
    after = _table_with(app, "generation_kwh")[["day", "hour", "microgrid_id", "generation_kwh", "demand_kwh"]]
    assert not app.exception
    pd.testing.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True))
    assert _costs(app)[0] != "8,633.60"  # the mechanisms did re-run with the new tariff


def test_24h_sample_flags_missing_equilibria(app):
    app.sidebar.selectbox(key="scenario").set_value(SAMPLE).run()
    assert not app.exception
    assert _costs(app) == ["8,633.60", "7,373.66", "7,771.52"]
    assert any("NO PURE EQUILIBRIUM FOUND in 2 interval(s)" in e.value for e in app.error)
    app.selectbox(key="game_interval").set_value((0, 12)).run()
    assert any("is NOT NASH" in e.value for e in app.error)
    app.selectbox(key="game_interval").set_value((0, 9)).run()
    assert any("Verified pure Nash equilibrium: MG1=7.50, MG3=7.50" in s.value for s in app.success)


def test_six_microgrid_day_reports_the_enumeration_cap(app):
    app.sidebar.selectbox(key="scenario").set_value(CUSTOM).run()
    app.sidebar.selectbox(key="microgrids").set_value(6).run()
    assert not app.exception
    assert _metric(app, "Microgrids") == "6"
    assert any("max_competitive_sellers" in w.value for w in app.warning)


def test_inverted_tariffs_show_an_economic_warning(app):
    app.sidebar.slider(key="retail").set_value(4.0).run()
    app.sidebar.slider(key="export").set_value(6.0).run()
    assert not app.exception
    assert any("retail price is below export price" in w.value for w in app.warning)
    assert _costs(app)[1] == _costs(app)[0]  # no P2P trade can have positive surplus


def test_solver_failure_is_reported_without_numbers(repo_root, monkeypatch):
    import microgrid.cooperative as cooperative
    from scipy.optimize import OptimizeResult

    def infeasible(c, **kwargs):
        return OptimizeResult(x=None, status=2, success=False, message="The problem is infeasible.", fun=None)

    monkeypatch.setattr(cooperative, "linprog", infeasible)
    app = testing.AppTest.from_file(str(repo_root / "app.py"), default_timeout=120)
    app.run()
    app.sidebar.slider(key="alpha").set_value(0.35).run()  # a fresh parameter set, so nothing comes from the cache
    assert not app.exception
    assert any("could not produce a validated result" in e.value for e in app.error)
    assert not any(m.label.startswith("Cooperative") for m in app.metric)
