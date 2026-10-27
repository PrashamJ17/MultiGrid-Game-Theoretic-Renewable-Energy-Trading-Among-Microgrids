"""Independent audit of exported results (blueprint Prompt 6: "check exported CSVs").

Usage (from the repository root, after `python -m microgrid.cli experiment E1 --out-dir results/experiments`):

    python scripts/audit_exports.py results/experiments/E1

This script deliberately does NOT import the ``microgrid`` package. It re-derives every accounting identity and
every equilibrium label from the exported CSV/JSON files with plain pandas, so a bug in the package cannot hide
itself. Exit status 0 = every check passed, 1 = at least one check failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

KEY = ["day", "hour", "microgrid_id"]


class Audit:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str) -> None:
        self.rows.append((name, bool(ok), detail))

    def report(self) -> int:
        width = max(len(name) for name, _, _ in self.rows)
        for name, ok, detail in self.rows:
            print(f"[{'PASS' if ok else 'FAIL'}] {name:<{width}}  {detail}")
        failed = sum(1 for _, ok, _ in self.rows if not ok)
        print(f"\n{len(self.rows) - failed} passed, {failed} failed")
        return 1 if failed else 0


def audit(folder: Path) -> int:
    a = Audit()
    configs = sorted((folder / "configs").glob("*.json"))
    cfg = json.loads(configs[0].read_text())
    b, x_price = cfg["retail_price_inr_per_kwh"], cfg["export_price_inr_per_kwh"]
    f_s, f_b, tol = cfg["seller_fee_inr_per_kwh"], cfg["buyer_fee_inr_per_kwh"], cfg["tol"]
    money_tol = 1e-6

    profiles = pd.read_csv(folder / "profiles.csv")
    settlement = pd.read_csv(folder / "settlement.csv")
    trades = pd.read_csv(folder / "trades.csv")
    diagnostics = pd.read_csv(folder / "diagnostics.csv")
    curves = pd.read_csv(folder / "payoff_curves.csv")

    # --- inputs ---------------------------------------------------------------
    expected_rows = cfg["days"] * cfg["hours_per_day"] * len(cfg["microgrid_ids"])
    a.check("profiles: complete unique grid", len(profiles) == expected_rows and not profiles.duplicated(KEY).any(),
            f"{len(profiles)} rows, expected {expected_rows}")
    a.check("profiles: non-negative finite energy",
            bool((profiles[["generation_kwh", "demand_kwh"]] >= 0).all().all()), "generation and demand >= 0")

    p = profiles.assign(S=(profiles.generation_kwh - profiles.demand_kwh).clip(lower=0),
                        R=(profiles.demand_kwh - profiles.generation_kwh).clip(lower=0))
    s = settlement.merge(p, on=KEY, how="left", validate="many_to_one")
    a.check("settlement covers every profile row per mechanism",
            s.groupby("mechanism").size().eq(len(profiles)).all() and s.generation_kwh.notna().all(),
            f"{settlement.mechanism.nunique()} mechanisms x {len(profiles)} rows")

    # --- physics ----------------------------------------------------------------
    residual = (s.generation_kwh + s.grid_import_kwh + s.p2p_import_kwh) - (s.demand_kwh + s.grid_export_kwh + s.p2p_export_kwh)
    a.check("energy identity G + B + in = D + X + out", residual.abs().max() <= 1e-6,
            f"max |residual| = {residual.abs().max():.2e} kWh")
    a.check("no simultaneous grid import and export",
            not ((s.grid_import_kwh > tol) & (s.grid_export_kwh > tol)).any(), "per row")
    a.check("no simultaneous P2P buying and selling",
            not ((s.p2p_import_kwh > tol) & (s.p2p_export_kwh > tol)).any(), "per row")

    # --- trades vs settlement, capacities --------------------------------------
    bought = trades.groupby(["mechanism", "day", "hour", "buyer_id"]).quantity_kwh.sum()
    sold = trades.groupby(["mechanism", "day", "hour", "seller_id"]).quantity_kwh.sum()
    idx = s.set_index(["mechanism", "day", "hour", "microgrid_id"])
    in_from_trades = bought.reindex(idx.index, fill_value=0.0)
    out_from_trades = sold.reindex(idx.index, fill_value=0.0)
    a.check("settlement P2P flows equal the trade log",
            (idx.p2p_import_kwh - in_from_trades.values).abs().max() <= 1e-9
            and (idx.p2p_export_kwh - out_from_trades.values).abs().max() <= 1e-9, "per mechanism, hour, microgrid")
    a.check("sellers never exceed surplus, buyers never exceed deficit",
            bool((idx.p2p_export_kwh <= idx.S + tol).all() and (idx.p2p_import_kwh <= idx.R + tol).all()), "per row")
    a.check("baseline mechanism has no trades and B = R, X = S",
            trades[trades.mechanism == "baseline"].empty
            and bool(((s[s.mechanism == "baseline"].grid_import_kwh - s[s.mechanism == "baseline"].R).abs() <= 1e-9).all()),
            "utility-only")

    # --- money -------------------------------------------------------------------
    base_formula = b * s.R - x_price * s.S
    a.check("baseline cost = b*R - a*S in every row", (s.baseline_cost_inr - base_formula).abs().max() <= money_tol,
            f"max deviation {(s.baseline_cost_inr - base_formula).abs().max():.2e} INR")
    a.check("gain = baseline cost - net cost", (s.gain_inr - (s.baseline_cost_inr - s.p2p_cost_inr)).abs().max() <= money_tol,
            "per row")
    hourly = s.groupby(["mechanism", "day", "hour"]).agg(cost=("p2p_cost_inr", "sum"), B=("grid_import_kwh", "sum"),
                                                          X=("grid_export_kwh", "sum"), gain=("gain_inr", "sum"))
    fees = trades.assign(fee=trades.quantity_kwh * (trades.seller_fee_inr_per_kwh + trades.buyer_fee_inr_per_kwh),
                         spread=trades.quantity_kwh * (b - x_price - trades.seller_fee_inr_per_kwh - trades.buyer_fee_inr_per_kwh))
    by_hour = fees.groupby(["mechanism", "day", "hour"])[["fee", "spread"]].sum().reindex(hourly.index, fill_value=0.0)
    external = b * hourly.B - x_price * hourly.X + by_hour.fee
    a.check("internal P2P payments cancel: community cost = grid bill + fees",
            (hourly.cost - external).abs().max() <= money_tol, f"max deviation {(hourly.cost - external).abs().max():.2e} INR")
    a.check("community gain = sum of q*(b - a - f_s - f_b)", (hourly.gain - by_hour.spread).abs().max() <= money_tol,
            "per mechanism and hour (no double-counted fees)")
    a.check("voluntary participation: no negative gain", bool((s.gain_inr >= -money_tol).all()),
            f"min gain {s.gain_inr.min():.4f} INR")
    a.check("voluntary prices: buyer pays < retail, seller nets >= export",
            bool(((trades.price_inr_per_kwh + trades.buyer_fee_inr_per_kwh) < b - tol).all()
                 and ((trades.price_inr_per_kwh - trades.seller_fee_inr_per_kwh) >= x_price - tol).all()), "every trade")
    a.check("fees recorded on trades match the config",
            bool((trades.seller_fee_inr_per_kwh == f_s).all() and (trades.buyer_fee_inr_per_kwh == f_b).all()), "flat fees")

    # --- equilibrium labels --------------------------------------------------------
    comp = diagnostics[diagnostics.mechanism == "competitive"]
    success = comp[comp.status == "success"]
    no_ne = comp[comp.status == "NO_PURE_NE"]
    a.check("status success => pure NE found, >= 1 equilibrium, regret <= tol, a selected profile",
            bool(success.pure_ne_found.astype(str).eq("True").all() and (success.equilibrium_count >= 1).all()
                 and (success.max_unilateral_regret_inr <= tol).all() and success.selected_profile.notna().all()),
            f"{len(success)} intervals")
    a.check("NO_PURE_NE => no NE claimed, zero equilibria, regret > tol, NOT NASH profile, no trades",
            bool(no_ne.pure_ne_found.astype(str).eq("False").all() and (no_ne.equilibrium_count == 0).all()
                 and (no_ne.max_unilateral_regret_inr > tol).all() and no_ne.not_nash_profile.notna().all()
                 and no_ne.p2p_volume_kwh.eq(0).all()),
            f"{len(no_ne)} intervals")
    priced = trades[trades.mechanism == "competitive"].merge(success[["day", "hour", "selected_profile"]], on=["day", "hour"])
    matches = all(abs(json.loads(r.selected_profile)[r.seller_id] - r.price_inr_per_kwh) <= 1e-9 for r in priced.itertuples())
    a.check("competitive trades are priced at the selected equilibrium", matches and len(priced) == len(
        trades[trades.mechanism == "competitive"]), f"{len(priced)} trades")
    other = diagnostics[diagnostics.mechanism != "competitive"]
    a.check("baseline and cooperative never carry an equilibrium label", other.pure_ne_found.isna().all(), "pure_ne_found empty")

    # Independent best-response check from the exported unilateral-deviation curves.
    ne_ok, not_nash_ok = True, True
    for (hour, is_nash), group in curves.groupby(["hour", "held_profile_is_nash"]):
        gains = []
        for _, seller in group.groupby("seller"):
            held = seller[seller.own_price_inr_per_kwh == seller.held_price_inr_per_kwh.iloc[0]].utility_inr.iloc[0]
            gains.append(seller.utility_inr.max() - held)
        if is_nash and max(gains) > tol:
            ne_ok = False
        if not is_nash and max(gains) <= tol:
            not_nash_ok = False
    a.check("every settled NE: no seller gains by any unilateral price change", ne_ok,
            f"{curves[curves.held_profile_is_nash].hour.nunique()} intervals re-checked from payoff_curves.csv")
    a.check("every NOT NASH profile: some seller strictly gains by deviating", not_nash_ok,
            f"{curves[~curves.held_profile_is_nash].hour.nunique()} intervals re-checked")
    return a.report()


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent audit of exported E1 results")
    parser.add_argument("folder", type=Path, help="experiment output folder containing E1-style exports")
    sys.exit(audit(parser.parse_args().folder))


if __name__ == "__main__":
    main()
