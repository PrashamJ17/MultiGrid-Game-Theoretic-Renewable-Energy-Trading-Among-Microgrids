"""Shared financial settlement and market validation.

``settle_market`` is the single settlement function used by every mechanism
(baseline, cooperative and, in Phase 3, competitive). A mechanism only
decides *which* trades happen and at *what* price. Every quantity and every
rupee is then derived here in exactly the same way.

Accounting for player i in one interval (b = retail price, a = export price
of that hour; f_s, f_b = per-kWh fees recorded on each trade):

    P2P_in   = sum_j q[j,i]                P2P_out = sum_j q[i,j]
    B        = R - P2P_in                  X       = S - P2P_out      (residual grid flows)
    purchase = sum_j p[j,i] * q[j,i]       sales   = sum_j p[i,j] * q[i,j]
    fees     = sum_j f_b * q[j,i] + sum_j f_s * q[i,j]
    C        = b*B - a*X + purchase - sales + fees                   (signed net cost)
    C_base   = b*R - a*S                   U       = C_base - C       (gain vs baseline)

Community identities, checked by ``validate_market``:

    sum_i purchase_i = sum_i sales_i                 (internal payments cancel)
    sum_i fees_i     = sum_trades q * (f_s + f_b)    (operator fee revenue)
    sum_i U_i        = sum_trades q * (b - a - f_s - f_b)

Interaction
    Uses ``energy`` for balance/tolerance checks and ``models`` for types.
    ``baseline`` calls ``settle_market`` with no trades; ``cooperative``
    calls it with the bargained trades. ``validate_market`` runs inside
    every settlement, so only validated results reach the reporting layer.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import pandas as pd

from .data import results_to_settlement_frame
from .energy import check_player_balance, clamp_tol
from .models import (
    EnergyBalanceError,
    HourState,
    MarketDiagnostics,
    MarketResult,
    PlayerSettlement,
    Role,
    ScenarioConfig,
    SettlementError,
    TariffSchedule,
    Trade,
)

_MONEY_FIELDS = (
    "grid_purchase_inr",
    "grid_export_revenue_inr",
    "p2p_purchase_inr",
    "p2p_sales_inr",
    "fees_inr",
    "baseline_cost_inr",
)


def baseline_cost(
    surplus_kwh: float,
    deficit_kwh: float,
    retail_price_inr_per_kwh: float,
    export_price_inr_per_kwh: float,
) -> float:
    """Signed utility-only cost of one microgrid in one interval: b*R - a*S (INR).

    Positive means the microgrid pays. Negative means it earns export income.
    """
    return retail_price_inr_per_kwh * deficit_kwh - export_price_inr_per_kwh * surplus_kwh


def _tariffs(cfg: ScenarioConfig, tariffs: TariffSchedule | None) -> TariffSchedule:
    return tariffs if tariffs is not None else TariffSchedule.from_config(cfg)


def _check_trades(state: HourState, trades: Sequence[Trade], tol: float) -> None:
    """Structural trade checks: right interval, known and eligible traders, no phantom or duplicate trades."""
    roles = {p.microgrid_id: p.role for p in state.players}
    seen: set[tuple[str, str]] = set()
    for t in trades:
        where = f"trade {t.seller_id}->{t.buyer_id}"
        if (t.day, t.hour) != (state.day, state.hour):
            raise SettlementError(
                f"{where} is for day {t.day} hour {t.hour}, not this interval "
                f"(day {state.day} hour {state.hour})"
            )
        for mg in (t.seller_id, t.buyer_id):
            if mg not in roles:
                raise SettlementError(f"{where}: unknown microgrid {mg!r}")
        if roles[t.seller_id] is not Role.SELLER:
            raise SettlementError(f"{where}: {t.seller_id} has no surplus above tolerance and cannot sell")
        if roles[t.buyer_id] is not Role.BUYER:
            raise SettlementError(f"{where}: {t.buyer_id} has no deficit above tolerance and cannot buy")
        if t.quantity_kwh <= tol:
            raise SettlementError(
                f"{where}: quantity {t.quantity_kwh} kWh is not above tolerance {tol}; "
                "phantom trades are not allowed"
            )
        pair = (t.seller_id, t.buyer_id)
        if pair in seen:
            raise SettlementError(f"{where}: more than one trade for the same seller-buyer pair")
        seen.add(pair)


def _check_individual_rationality(trades: Sequence[Trade], retail: float, export: float, tol: float) -> None:
    """Voluntary trade at the price level.

    The buyer's delivered price must be strictly below retail (by tol), and
    the seller's net receipt must be at least the export price.
    """
    for t in trades:
        delivered = t.buyer_delivered_price_inr_per_kwh
        if delivered > retail - tol:
            raise SettlementError(
                f"trade {t.seller_id}->{t.buyer_id}: buyer delivered price {delivered} INR/kWh is not "
                f"below the retail price {retail} INR/kWh, so the buyer would not trade voluntarily"
            )
        net = t.seller_net_price_inr_per_kwh
        if net < export - tol:
            raise SettlementError(
                f"trade {t.seller_id}->{t.buyer_id}: seller net receipt {net} INR/kWh is below the "
                f"export price {export} INR/kWh, so the seller would not trade voluntarily"
            )


def settle_market(
    state: HourState,
    trades: Iterable[Trade],
    cfg: ScenarioConfig,
    *,
    mechanism: str,
    tariffs: TariffSchedule | None = None,
    diagnostics: MarketDiagnostics | None = None,
) -> MarketResult:
    """Settle one interval's trades and return a validated ``MarketResult``.

    Raises ``SettlementError`` for invalid trades and ``EnergyBalanceError``
    when trades exceed a seller's surplus or a buyer's deficit. No result is
    returned unless ``validate_market`` passes.
    """
    schedule = _tariffs(cfg, tariffs)
    retail = schedule.retail_at(state.hour)
    export = schedule.export_at(state.hour)
    ordered = tuple(sorted(trades, key=lambda t: (t.seller_id, t.buyer_id)))
    _check_trades(state, ordered, cfg.tol)
    _check_individual_rationality(ordered, retail, export, cfg.tol)

    settlements = []
    for player in state.players:
        mg = player.microgrid_id
        bought = [t for t in ordered if t.buyer_id == mg]
        sold = [t for t in ordered if t.seller_id == mg]
        p2p_in = math.fsum(t.quantity_kwh for t in bought)
        p2p_out = math.fsum(t.quantity_kwh for t in sold)
        grid_import = clamp_tol(player.deficit_kwh - p2p_in, cfg.tol, name=f"{mg} residual grid import (deficit - P2P purchases)")
        grid_export = clamp_tol(player.surplus_kwh - p2p_out, cfg.tol, name=f"{mg} residual grid export (surplus - P2P sales)")
        purchase = math.fsum(t.payment_inr for t in bought)
        sales = math.fsum(t.payment_inr for t in sold)
        fees = math.fsum(
            [t.quantity_kwh * t.buyer_fee_inr_per_kwh for t in bought]
            + [t.quantity_kwh * t.seller_fee_inr_per_kwh for t in sold]
        )
        grid_purchase = retail * grid_import
        export_revenue = export * grid_export
        base = baseline_cost(player.surplus_kwh, player.deficit_kwh, retail, export)
        cost = math.fsum([grid_purchase, -export_revenue, purchase, -sales, fees])
        settlements.append(
            PlayerSettlement(
                scenario_id=cfg.scenario_id,
                mechanism=mechanism,
                day=state.day,
                hour=state.hour,
                microgrid_id=mg,
                grid_import_kwh=grid_import,
                grid_export_kwh=grid_export,
                p2p_import_kwh=p2p_in,
                p2p_export_kwh=p2p_out,
                grid_purchase_inr=grid_purchase,
                grid_export_revenue_inr=export_revenue,
                p2p_purchase_inr=purchase,
                p2p_sales_inr=sales,
                fees_inr=fees,
                baseline_cost_inr=base,
                p2p_cost_inr=cost,
                gain_inr=base - cost,
            )
        )

    result = MarketResult(
        scenario_id=cfg.scenario_id,
        mechanism=mechanism,
        day=state.day,
        hour=state.hour,
        trades=ordered,
        settlements=tuple(settlements),
        diagnostics=diagnostics
        if diagnostics is not None
        else MarketDiagnostics(status="success" if ordered else "no_trade"),
    )
    validate_market(state, result, cfg, tariffs=schedule)
    return result


def validate_market(
    state: HourState,
    result: MarketResult,
    cfg: ScenarioConfig,
    *,
    tariffs: TariffSchedule | None = None,
    require_individual_rationality: bool = True,
) -> None:
    """Re-derive every quantity of ``result`` from its trade log and the tariffs.

    Raises on the first inconsistency.

    Checks:
        * settlements cover exactly the interval's players, and trades are structurally valid
        * P2P flows match the trade log; sellers never exceed their surplus and
          buyers never exceed their deficit (``EnergyBalanceError``)
        * each player's energy identity holds, with no simultaneous import/export
        * each money component, net cost and gain matches its formula
        * P2P payments cancel in aggregate, and fees equal operator revenue
        * community gain = sum over trades of q * (b - a - f_s - f_b)
        * if ``require_individual_rationality``: voluntary prices, and no
          player worse off than baseline
    """
    tol = cfg.tol
    schedule = _tariffs(cfg, tariffs)
    retail = schedule.retail_at(state.hour)
    export = schedule.export_at(state.hour)

    if (result.day, result.hour) != (state.day, state.hour):
        raise SettlementError(
            f"result is for day {result.day} hour {result.hour}, state is day {state.day} hour {state.hour}"
        )
    covered = tuple(s.microgrid_id for s in result.settlements)
    if covered != state.microgrid_ids:
        raise SettlementError(f"settlements cover {covered} but the interval has players {state.microgrid_ids}")
    _check_trades(state, result.trades, tol)
    if require_individual_rationality:
        _check_individual_rationality(result.trades, retail, export, tol)

    money_scale = max(
        1.0, math.fsum(abs(getattr(s, name)) for s in result.settlements for name in _MONEY_FIELDS)
    )
    money_tol = tol * money_scale

    for s in result.settlements:
        mg = s.microgrid_id
        player = state.player(mg)
        bought = [t for t in result.trades if t.buyer_id == mg]
        sold = [t for t in result.trades if t.seller_id == mg]
        p2p_in = math.fsum(t.quantity_kwh for t in bought)
        p2p_out = math.fsum(t.quantity_kwh for t in sold)
        if abs(s.p2p_import_kwh - p2p_in) > tol or abs(s.p2p_export_kwh - p2p_out) > tol:
            raise SettlementError(
                f"{mg}: settlement P2P flows ({s.p2p_import_kwh} in, {s.p2p_export_kwh} out) do not match "
                f"the trade log ({p2p_in} in, {p2p_out} out)"
            )
        if p2p_out > player.surplus_kwh + tol:
            raise EnergyBalanceError(f"{mg} sells {p2p_out} kWh but has only {player.surplus_kwh} kWh surplus")
        if p2p_in > player.deficit_kwh + tol:
            raise EnergyBalanceError(f"{mg} buys {p2p_in} kWh but needs only {player.deficit_kwh} kWh")
        check_player_balance(
            microgrid_id=mg,
            generation_kwh=player.generation_kwh,
            demand_kwh=player.demand_kwh,
            grid_import_kwh=s.grid_import_kwh,
            grid_export_kwh=s.grid_export_kwh,
            p2p_import_kwh=s.p2p_import_kwh,
            p2p_export_kwh=s.p2p_export_kwh,
            tol=tol,
        )

        expected = {
            "grid_purchase_inr": retail * s.grid_import_kwh,
            "grid_export_revenue_inr": export * s.grid_export_kwh,
            "p2p_purchase_inr": math.fsum(t.payment_inr for t in bought),
            "p2p_sales_inr": math.fsum(t.payment_inr for t in sold),
            "fees_inr": math.fsum(
                [t.quantity_kwh * t.buyer_fee_inr_per_kwh for t in bought]
                + [t.quantity_kwh * t.seller_fee_inr_per_kwh for t in sold]
            ),
            "baseline_cost_inr": baseline_cost(player.surplus_kwh, player.deficit_kwh, retail, export),
        }
        for name, value in expected.items():
            if abs(getattr(s, name) - value) > money_tol:
                raise SettlementError(f"{mg}: {name} = {getattr(s, name)} but trades and tariffs imply {value}")
        cost = math.fsum(
            [s.grid_purchase_inr, -s.grid_export_revenue_inr, s.p2p_purchase_inr, -s.p2p_sales_inr, s.fees_inr]
        )
        if abs(s.p2p_cost_inr - cost) > money_tol:
            raise SettlementError(f"{mg}: net cost {s.p2p_cost_inr} does not equal its components ({cost})")
        if abs(s.gain_inr - (s.baseline_cost_inr - s.p2p_cost_inr)) > money_tol:
            raise SettlementError(f"{mg}: gain {s.gain_inr} is not baseline cost minus net cost")
        if require_individual_rationality and s.gain_inr < -money_tol:
            raise SettlementError(
                f"{mg}: gain {s.gain_inr} INR is negative, so the player would be worse off than with "
                "utility-only trading (voluntary participation violated)"
            )

    purchases = math.fsum(s.p2p_purchase_inr for s in result.settlements)
    sales = math.fsum(s.p2p_sales_inr for s in result.settlements)
    if abs(purchases - sales) > money_tol:
        raise SettlementError(f"P2P payments do not cancel: purchases {purchases} INR vs sales {sales} INR")
    fees = math.fsum(s.fees_inr for s in result.settlements)
    if abs(fees - result.total_fee_revenue_inr) > money_tol:
        raise SettlementError(f"fees charged {fees} INR differ from operator fee revenue {result.total_fee_revenue_inr} INR")
    spread_gain = math.fsum(
        t.quantity_kwh * (retail - export - t.seller_fee_inr_per_kwh - t.buyer_fee_inr_per_kwh)
        for t in result.trades
    )
    if abs(result.total_gain_inr - spread_gain) > money_tol:
        raise SettlementError(
            f"community gain {result.total_gain_inr} INR differs from the avoided grid spread minus fees "
            f"({spread_gain} INR)"
        )


def summarize_settlements(results: Sequence[MarketResult]) -> pd.DataFrame:
    """Per-microgrid totals over all intervals (kWh and signed INR)."""
    columns = [
        "microgrid_id",
        "p2p_export_kwh",
        "p2p_import_kwh",
        "grid_import_kwh",
        "grid_export_kwh",
        "baseline_cost_inr",
        "p2p_cost_inr",
        "gain_inr",
    ]
    frame = results_to_settlement_frame(results)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    return frame.groupby("microgrid_id", sort=True)[columns[1:]].sum().reset_index().loc[:, columns]
