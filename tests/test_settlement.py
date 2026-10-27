"""Shared settlement and market validation (blueprint section 9; T07-T10).

The trades here are written by hand, copied from the blueprint table, so the
accounting is tested independently of the LP that normally produces them.
"""

from __future__ import annotations

import dataclasses

import pytest

from microgrid.baseline import solve_baseline
from microgrid.models import EnergyBalanceError, SettlementError, TariffSchedule, Trade
from microgrid.settlement import settle_market, summarize_settlements, validate_market

GOLDEN_TRADES = [("MG1", "MG2", 15.0), ("MG1", "MG4", 5.0), ("MG3", "MG4", 10.0)]


def _trades(price=5.5, f_s=0.0, f_b=0.0, rows=GOLDEN_TRADES, hour=12):
    return [Trade(0, hour, s, b, q, price, f_s, f_b) for s, b, q in rows]


def _settle(state, cfg, trades, mechanism="cooperative", **kwargs):
    return settle_market(state, trades, cfg, mechanism=mechanism, **kwargs)


def test_empty_trade_log_reproduces_the_baseline(golden_state, golden_cfg):
    settled = _settle(golden_state, golden_cfg, (), mechanism="baseline")
    assert settled.settlements == solve_baseline(golden_state, golden_cfg).settlements
    assert settled.total_cost_inr == pytest.approx(230.0)
    assert settled.total_gain_inr == 0.0
    assert settled.diagnostics.status == "no_trade"


def test_blueprint_trades_settle_to_blueprint_values(golden_state, golden_cfg):
    result = _settle(golden_state, golden_cfg, _trades())
    #          grid_in grid_out p2p_in p2p_out  cost    gain
    expected = {
        "MG1": (0.0, 0.0, 0.0, 20.0, -110.0, 50.0),
        "MG2": (0.0, 0.0, 15.0, 0.0, 82.5, 37.5),
        "MG3": (0.0, 0.0, 0.0, 10.0, -55.0, 25.0),
        "MG4": (10.0, 0.0, 15.0, 0.0, 162.5, 37.5),
    }
    for mg, values in expected.items():
        s = result.settlement_for(mg)
        actual = (s.grid_import_kwh, s.grid_export_kwh, s.p2p_import_kwh, s.p2p_export_kwh, s.p2p_cost_inr, s.gain_inr)
        assert actual == pytest.approx(values, abs=1e-9), mg
    assert result.total_cost_inr == pytest.approx(80.0)
    assert result.total_gain_inr == pytest.approx(150.0)
    assert result.total_p2p_volume_kwh == pytest.approx(30.0)
    assert result.diagnostics.status == "success"


def test_internal_payments_cancel(golden_state, golden_cfg):  # T09
    result = _settle(golden_state, golden_cfg, _trades())
    purchases = sum(s.p2p_purchase_inr for s in result.settlements)
    sales = sum(s.p2p_sales_inr for s in result.settlements)
    assert purchases == pytest.approx(sales, abs=1e-9)
    assert purchases == pytest.approx(30 * 5.5)
    assert result.total_p2p_payment_inr == pytest.approx(165.0)


def test_fee_bookkeeping(golden_state, golden_cfg):
    cfg = dataclasses.replace(golden_cfg, seller_fee_inr_per_kwh=1.0, buyer_fee_inr_per_kwh=1.0)
    result = _settle(golden_state, cfg, _trades(f_s=1.0, f_b=1.0))
    assert result.total_fee_revenue_inr == pytest.approx(60.0)
    assert sum(s.fees_inr for s in result.settlements) == pytest.approx(60.0)
    assert result.settlement_for("MG1").fees_inr == pytest.approx(20.0)  # seller fee on 20 kWh
    assert result.settlement_for("MG4").fees_inr == pytest.approx(15.0)  # buyer fee on 15 kWh
    assert result.settlement_for("MG1").p2p_cost_inr == pytest.approx(-90.0)
    assert result.total_gain_inr == pytest.approx(90.0)
    assert result.total_cost_inr == pytest.approx(140.0)


@pytest.mark.parametrize("price", [3.5, 5.5, 7.5])
def test_price_only_redistributes_the_community_gain(golden_state, golden_cfg, price):
    result = _settle(golden_state, golden_cfg, _trades(price=price))
    assert result.total_gain_inr == pytest.approx(30 * (8.0 - 3.0))
    sellers_gain = result.settlement_for("MG1").gain_inr + result.settlement_for("MG3").gain_inr
    assert sellers_gain == pytest.approx(30 * (price - 3.0))


def test_energy_identity_with_p2p(golden_state, golden_cfg):  # T10
    result = _settle(golden_state, golden_cfg, _trades())
    for s in result.settlements:
        p = golden_state.player(s.microgrid_id)
        supply = p.generation_kwh + s.grid_import_kwh + s.p2p_import_kwh
        use = p.demand_kwh + s.grid_export_kwh + s.p2p_export_kwh
        assert supply == pytest.approx(use, abs=1e-9), s.microgrid_id


def test_hourly_tariffs_flow_into_settlement(golden_state, golden_cfg):
    retail, export = [8.0] * 24, [3.0] * 24
    retail[12], export[12] = 10.0, 2.0
    result = _settle(golden_state, golden_cfg, _trades(), tariffs=TariffSchedule(tuple(retail), tuple(export)))
    assert result.settlement_for("MG4").p2p_cost_inr == pytest.approx(10 * 10.0 + 15 * 5.5)
    assert result.total_gain_inr == pytest.approx(30 * (10.0 - 2.0))


def test_seller_cannot_sell_more_than_its_surplus(golden_state, golden_cfg):  # T08
    with pytest.raises(EnergyBalanceError, match="MG1"):
        _settle(golden_state, golden_cfg, _trades(rows=[("MG1", "MG2", 15.0), ("MG1", "MG4", 10.0)]))


def test_buyer_cannot_buy_more_than_its_deficit(golden_state, golden_cfg):
    with pytest.raises(EnergyBalanceError, match="MG2"):
        _settle(golden_state, golden_cfg, _trades(rows=[("MG1", "MG2", 16.0)]))


@pytest.mark.parametrize(
    "trade,match",
    [
        (Trade(0, 12, "MG2", "MG4", 1.0, 5.5), "cannot sell"),
        (Trade(0, 12, "MG1", "MG3", 1.0, 5.5), "cannot buy"),
        (Trade(0, 12, "MG1", "MG9", 1.0, 5.5), "unknown microgrid"),
        (Trade(0, 13, "MG1", "MG2", 1.0, 5.5), "not this interval"),
        (Trade(0, 12, "MG1", "MG2", 0.0, 5.5), "phantom"),
        (Trade(0, 12, "MG1", "MG2", 1e-9, 5.5), "phantom"),
        (Trade(0, 12, "MG1", "MG2", 1.0, 8.0), "not below the retail price"),
        (Trade(0, 12, "MG1", "MG2", 1.0, 7.5, 0.0, 0.5), "not below the retail price"),
        (Trade(0, 12, "MG1", "MG2", 1.0, 2.9), "below the export price"),
        (Trade(0, 12, "MG1", "MG2", 1.0, 3.5, 1.0, 0.0), "below the export price"),
    ],
)
def test_invalid_trades_are_rejected(golden_state, golden_cfg, trade, match):
    with pytest.raises(SettlementError, match=match):
        _settle(golden_state, golden_cfg, [trade])


def test_duplicate_seller_buyer_pair_is_rejected(golden_state, golden_cfg):
    with pytest.raises(SettlementError, match="same seller-buyer pair"):
        _settle(golden_state, golden_cfg, _trades(rows=[("MG1", "MG2", 5.0), ("MG1", "MG2", 5.0)]))


def test_validate_market_detects_tampered_results(golden_state, golden_cfg):
    result = _settle(golden_state, golden_cfg, _trades())
    validate_market(golden_state, result, golden_cfg)  # the genuine result passes
    s0 = result.settlements[0]
    rest = result.settlements[1:]

    shifted_cost = dataclasses.replace(s0, p2p_cost_inr=s0.p2p_cost_inr - 1.0, gain_inr=s0.gain_inr + 1.0)
    with pytest.raises(SettlementError, match="does not equal its components"):
        validate_market(golden_state, dataclasses.replace(result, settlements=(shifted_cost,) + rest), golden_cfg)

    inflated_sales = dataclasses.replace(
        s0, p2p_sales_inr=s0.p2p_sales_inr + 5.0, p2p_cost_inr=s0.p2p_cost_inr - 5.0, gain_inr=s0.gain_inr + 5.0
    )
    with pytest.raises(SettlementError, match="p2p_sales_inr"):
        validate_market(golden_state, dataclasses.replace(result, settlements=(inflated_sales,) + rest), golden_cfg)

    with pytest.raises(SettlementError, match="do not match the trade log"):
        validate_market(golden_state, dataclasses.replace(result, trades=result.trades[:2]), golden_cfg)

    with pytest.raises(SettlementError, match="settlements cover"):
        validate_market(golden_state, dataclasses.replace(result, settlements=result.settlements[:3]), golden_cfg)


def test_summarize_settlements(golden_state, golden_cfg):
    summary = summarize_settlements([_settle(golden_state, golden_cfg, _trades())])
    assert summary.microgrid_id.tolist() == ["MG1", "MG2", "MG3", "MG4"]
    assert summary.set_index("microgrid_id").loc["MG1", "p2p_export_kwh"] == pytest.approx(20.0)
    assert summary.gain_inr.sum() == pytest.approx(150.0)
    assert summarize_settlements([]).empty
