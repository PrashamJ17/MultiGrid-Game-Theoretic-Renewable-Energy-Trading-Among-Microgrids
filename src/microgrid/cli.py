"""Command-line entry point: ``python -m microgrid.cli <command>``.

Commands
    generate     Write a reproducible synthetic profile CSV from a scenario config.
    baseline     Validate a profile CSV, run the utility-only baseline and report costs.
    cooperative  Validate a profile CSV, run the cooperative market (LP allocation +
                 bilateral Nash bargaining) and report trades, costs and gains.
    competitive  Validate a profile CSV, run the seller price game with exhaustive
                 pure-Nash verification and report equilibria, trades, costs and gains.
    compare      Run all three mechanisms on the same frozen profiles and report the
                 comparison metrics; --out-dir writes every table plus run information.

The CLI is a thin adapter. All calculations live in ``data``, ``energy``,
``settlement``, ``baseline``, ``cooperative`` and ``competitive``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

from .baseline import community_baseline_cost, run_baseline, summarize_baseline
from .competitive import STATUS_CAP_EXCEEDED, STATUS_NO_PURE_NE, run_competitive
from .cooperative import run_cooperative
from .experiments import SPECS, run_experiment, save_experiment
from .data import (
    diagnostics_records,
    generate_profiles,
    load_and_validate_profiles,
    load_config,
    load_tariffs_csv,
    results_to_settlement_frame,
    results_to_trade_frame,
    save_config,
    save_profiles,
    tariffs_to_frame,
    write_csv,
)
from .metrics import jain_fairness
from .models import (
    DataValidationError,
    EnergyBalanceError,
    MarketResult,
    ScenarioConfig,
    SettlementError,
    SolverError,
    TariffSchedule,
)
from .settlement import summarize_settlements
from .simulation import compare_all

MAX_PRINTED_TRADES = 30

#: (label, metric key, format) rows of the ``compare`` summary table.
COMPARE_ROWS = (
    ("Community net cost (INR)", "net_community_cost_inr", "{:,.2f}"),
    ("Gain vs baseline (INR)", "total_gain_inr", "{:,.2f}"),
    ("Cost reduction (%)", "cost_reduction_pct", "{:.2f}"),
    ("P2P volume (kWh)", "total_p2p_volume_kwh", "{:,.3f}"),
    ("Average P2P price (INR/kWh)", "average_price_inr_per_kwh", "{:.4f}"),
    ("Operator fees (INR)", "operator_fee_revenue_inr", "{:,.2f}"),
    ("Jain fairness of gains", "jain_fairness", "{:.6f}"),
    ("Efficiency vs LP optimum", "efficiency", "{:.4f}"),
    ("Trading intervals", "trading_intervals", "{}"),
    ("Pure-NE intervals", "pure_ne_intervals", "{}"),
    ("NO_PURE_NE intervals", "no_pure_ne_intervals", "{}"),
    ("Enumeration-cap intervals", "cap_exceeded_intervals", "{}"),
    ("Max regret at settled NE (INR)", "max_regret_inr", "{:.6f}"),
    ("Runtime (s)", "runtime_seconds", "{:.4f}"),
)


def _format_metric(value: object, template: str) -> str:
    return "N/A" if value is None else template.format(value)


def _add_market_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path, help="scenario config JSON")
    parser.add_argument("--profiles", required=True, type=Path, help="profile CSV")
    parser.add_argument(
        "--tariffs",
        type=Path,
        default=None,
        help="optional hourly tariff CSV (hour,retail_price_inr_per_kwh,export_price_inr_per_kwh)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="if given, write settlement/trade CSVs and the exact config used",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m microgrid.cli",
        description="MultiGrid tools: synthetic profiles, utility-only baseline, cooperative market.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="write a reproducible synthetic profile CSV")
    gen.add_argument("--config", required=True, type=Path, help="scenario config JSON")
    gen.add_argument("--out", required=True, type=Path, help="output CSV path")
    gen.add_argument("--seed", type=int, default=None, help="override the config seed")

    _add_market_arguments(sub.add_parser("baseline", help="run the utility-only baseline on a profile CSV"))
    _add_market_arguments(
        sub.add_parser("cooperative", help="run the cooperative market (LP + Nash bargaining) on a profile CSV")
    )
    _add_market_arguments(
        sub.add_parser("competitive", help="run the seller price game with pure-Nash verification on a profile CSV")
    )
    _add_market_arguments(
        sub.add_parser("compare", help="run all three mechanisms on the same profiles and compare them")
    )
    exp = sub.add_parser("experiment", help="run controlled experiments E1-E7 and save every table")
    exp.add_argument("experiments", nargs="+", help="experiment ids (E1 ... E7) or 'all'")
    exp.add_argument("--config", type=Path, default=Path("configs/default.json"), help="base scenario config")
    exp.add_argument("--profiles", type=Path, default=Path("data/sample_24h.csv"),
                     help="frozen profiles for experiments that do not regenerate them")
    exp.add_argument("--out-dir", type=Path, required=True, help="writes <out-dir>/<experiment id>/")
    return parser


def _cmd_generate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg = dataclasses.replace(cfg, seed=args.seed)
    profiles = generate_profiles(cfg)
    path = save_profiles(profiles, args.out)
    print(
        f"wrote {len(profiles)} rows ({cfg.days} day(s) x 24 hours x "
        f"{len(cfg.microgrid_ids)} microgrids, seed {cfg.seed}) to {path}"
    )
    return 0


def _load_market_inputs(args: argparse.Namespace) -> tuple[ScenarioConfig, TariffSchedule, str, pd.DataFrame]:
    cfg = load_config(args.config)
    if args.tariffs is not None:
        tariffs = load_tariffs_csv(args.tariffs)
        tariff_text = f"hourly tariffs from {args.tariffs}"
    else:
        tariffs = TariffSchedule.from_config(cfg)
        tariff_text = (
            f"flat tariffs: retail INR {cfg.retail_price_inr_per_kwh:.2f}/kWh, "
            f"export INR {cfg.export_price_inr_per_kwh:.2f}/kWh"
        )
    profiles = load_and_validate_profiles(args.profiles, cfg)
    return cfg, tariffs, tariff_text, profiles


def _write_outputs(
    args: argparse.Namespace,
    cfg: ScenarioConfig,
    tariffs: TariffSchedule,
    results: Sequence[MarketResult],
    mechanism: str,
) -> None:
    if args.out_dir is None:
        return
    out = args.out_dir
    stem = f"{cfg.scenario_id}_{mechanism}"
    written = [
        write_csv(results_to_settlement_frame(results), out / f"{stem}_settlement.csv"),
        write_csv(results_to_trade_frame(results), out / f"{stem}_trades.csv"),
        save_config(cfg, out / f"{cfg.scenario_id}_config.json"),
    ]
    if args.tariffs is not None:
        written.append(write_csv(tariffs_to_frame(tariffs), out / f"{cfg.scenario_id}_tariffs.csv"))
    print()
    for path in written:
        print(f"wrote {path}")


def _cmd_baseline(args: argparse.Namespace) -> int:
    cfg, tariffs, tariff_text, profiles = _load_market_inputs(args)
    results = run_baseline(profiles, cfg, tariffs)
    summary = summarize_baseline(results)
    total = community_baseline_cost(results)

    print(f"Scenario '{cfg.scenario_id}': utility-only baseline over {len(results)} interval(s)")
    print(f"Profiles: {args.profiles} ({len(profiles)} validated rows); {tariff_text}")
    print("Signed costs: positive = pays the utility, negative = net export income.")
    print()
    print(f"{'microgrid':<10} {'grid import kWh':>16} {'grid export kWh':>16} {'baseline cost INR':>18}")
    for row in summary.itertuples(index=False):
        print(
            f"{row.microgrid_id:<10} {row.grid_import_kwh:>16.3f} "
            f"{row.grid_export_kwh:>16.3f} {row.baseline_cost_inr:>18.2f}"
        )
    print()
    print(f"Community baseline net cost: INR {total:.2f}")
    _write_outputs(args, cfg, tariffs, results, "baseline")
    return 0


def _cmd_cooperative(args: argparse.Namespace) -> int:
    cfg, tariffs, tariff_text, profiles = _load_market_inputs(args)
    results = run_cooperative(profiles, cfg, tariffs)
    print(f"Scenario '{cfg.scenario_id}': cooperative market over {len(results)} interval(s)")
    print(
        "Allocation: welfare-maximising LP (SciPy HiGHS, lexicographic tie rule); "
        f"price: bilateral weighted Nash bargaining, alpha = {cfg.bargaining_seller_weight:.2f}"
    )
    _report_p2p(args, cfg, tariffs, tariff_text, profiles, results, "cooperative")
    return 0


def _profile_text(pairs) -> str:
    return ", ".join(f"{seller}={price:.2f}" for seller, price in pairs)


def _cmd_competitive(args: argparse.Namespace) -> int:
    cfg, tariffs, tariff_text, profiles = _load_market_inputs(args)
    results = run_competitive(profiles, cfg, tariffs)
    print(f"Scenario '{cfg.scenario_id}': competitive seller price game over {len(results)} interval(s)")
    print(
        f"Strategies: posted prices on a INR {cfg.price_step_inr:.2f} tick; buyers take the cheapest acceptable "
        "offer first (ties by seller ID); every unilateral deviation is checked"
    )
    print()
    games = [r for r in results if r.diagnostics.examined_profiles or r.diagnostics.status == STATUS_CAP_EXCEEDED]
    if games:
        print(f"{'day':>3} {'hour':>4}  {'status':<24} {'NE':>3} {'profiles':>8} {'max regret INR':>14}  prices")
        for r in games:
            d = r.diagnostics
            if d.selected_profile is not None:
                prices = _profile_text(d.selected_profile)
            elif d.not_nash_profile is not None:
                prices = f"NOT NASH (diagnostic): {_profile_text(d.not_nash_profile)}"
            else:
                prices = "-"
            count = "-" if d.equilibrium_count is None else d.equilibrium_count
            regret = "-" if d.max_regret_inr is None else f"{d.max_regret_inr:.4f}"
            print(f"{r.day:>3} {r.hour:>4}  {d.status:<24} {count:>3} {d.examined_profiles or 0:>8} {regret:>14}  {prices}")
        print()
    _report_p2p(args, cfg, tariffs, tariff_text, profiles, results, "competitive")
    no_ne = sum(1 for r in results if r.diagnostics.status == STATUS_NO_PURE_NE)
    capped = sum(1 for r in results if r.diagnostics.status == STATUS_CAP_EXCEEDED)
    if no_ne or capped:
        print(
            f"Note: {no_ne} interval(s) have NO PURE EQUILIBRIUM and {capped} exceeded the enumeration cap; "
            "they are settled without P2P trades, so the totals above are not equilibrium outcomes for those intervals."
        )
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    cfg, tariffs, tariff_text, profiles = _load_market_inputs(args)
    comparison = compare_all(profiles, cfg, tariffs)
    mechanisms = comparison.mechanisms
    metrics = comparison.metrics

    print(f"Scenario '{cfg.scenario_id}': all mechanisms on the same {len(profiles)} validated rows")
    print(
        f"{tariff_text}; fees: seller INR {cfg.seller_fee_inr_per_kwh:.2f}/kWh, buyer INR "
        f"{cfg.buyer_fee_inr_per_kwh:.2f}/kWh; alpha = {cfg.bargaining_seller_weight:.2f}; "
        f"price tick INR {cfg.price_step_inr:.2f}"
    )
    print(f"Baseline community net cost: INR {metrics[mechanisms[0]]['baseline_cost_inr']:,.2f}")
    print()
    print(f"{'metric':<32}" + "".join(f"{m:>16}" for m in mechanisms))
    for label, key, template in COMPARE_ROWS:
        print(f"{label:<32}" + "".join(f"{_format_metric(metrics[m][key], template):>16}" for m in mechanisms))
    print()
    print("Gain vs baseline per microgrid (INR):")
    print(f"{'microgrid':<32}" + "".join(f"{m:>16}" for m in mechanisms))
    for mg in sorted(cfg.microgrid_ids):
        print(f"{mg:<32}" + "".join(f"{metrics[m]['per_player_gain_inr'][mg]:>16,.2f}" for m in mechanisms))
    print()
    competitive = metrics.get("competitive")
    if competitive and (competitive["no_pure_ne_intervals"] or competitive["cap_exceeded_intervals"]):
        print(
            f"Note: {competitive['no_pure_ne_intervals']} interval(s) have NO PURE EQUILIBRIUM and "
            f"{competitive['cap_exceeded_intervals']} exceeded the enumeration cap; the competitive totals "
            "settle them without P2P trades."
        )
    print(
        "N/A = not applicable (no P2P volume, baseline cost <= 0, no LP optimum, all gains zero, "
        "or a metric defined only for the competitive game)."
    )

    if args.out_dir is not None:
        out = args.out_dir
        stem = f"{cfg.scenario_id}_compare"
        run_info = {
            "environment": dict(comparison.environment),
            "runtimes_seconds": dict(comparison.runtimes_seconds),
            "mechanisms": list(mechanisms),
        }
        info_path = out / f"{stem}_run_info.json"
        diagnostics_path = out / f"{stem}_diagnostics.json"
        out.mkdir(parents=True, exist_ok=True)
        info_path.write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
        diagnostics_path.write_text(
            json.dumps(diagnostics_records(r for _, rs in comparison.results for r in rs), indent=2) + "\n",
            encoding="utf-8",
        )
        written = [
            write_csv(comparison.summary_frame(), out / f"{stem}_summary.csv"),
            write_csv(comparison.player_frame(), out / f"{stem}_players.csv"),
            write_csv(comparison.trades_frame(), out / f"{stem}_trades.csv"),
            write_csv(comparison.settlement_frame(), out / f"{stem}_settlement.csv"),
            write_csv(comparison.diagnostics_frame(), out / f"{stem}_diagnostics.csv"),
            diagnostics_path,
            info_path,
            save_config(cfg, out / f"{cfg.scenario_id}_config.json"),
        ]
        if args.tariffs is not None:
            written.append(write_csv(tariffs_to_frame(tariffs), out / f"{cfg.scenario_id}_tariffs.csv"))
        print()
        for path in written:
            print(f"wrote {path}")
    return 0


def _cmd_experiment(args: argparse.Namespace) -> int:
    requested = [e.upper() for e in args.experiments]
    ids = sorted(SPECS) if "ALL" in requested else requested
    unknown = sorted(set(ids) - set(SPECS))
    if unknown:
        raise DataValidationError(f"unknown experiment(s) {unknown}; choose from {sorted(SPECS)} or 'all'")
    cfg = load_config(args.config)
    profiles = load_and_validate_profiles(args.profiles, cfg)
    for experiment_id in ids:
        spec = SPECS[experiment_id]
        result = run_experiment(experiment_id, cfg, profiles)
        paths = save_experiment(result, args.out_dir / experiment_id)
        runs = result.runs_frame()
        print(f"{experiment_id} {spec.title}: {spec.question}")
        print(
            f"{spec.variable:>22} {'base cost':>11} {'coop gain':>10} {'comp gain':>10} "
            f"{'coop kWh':>9} {'comp kWh':>9} {'NE h':>5} {'no-NE h':>8} {'cap h':>6}"
        )
        for _, group in runs.groupby("run_id", sort=False):
            row = group.set_index("mechanism")
            coop, comp = row.loc["cooperative"], row.loc["competitive"]
            print(
                f"{str(group[spec.variable].iloc[0]):>22} {row.loc['baseline', 'net_community_cost_inr']:>11,.2f} "
                f"{coop['total_gain_inr']:>10,.2f} {comp['total_gain_inr']:>10,.2f} "
                f"{coop['total_p2p_volume_kwh']:>9,.2f} {comp['total_p2p_volume_kwh']:>9,.2f} "
                f"{int(comp['pure_ne_intervals']):>5} {int(comp['no_pure_ne_intervals']):>8} "
                f"{int(comp['cap_exceeded_intervals']):>6}"
            )
        print(f"wrote {len(paths)} files to {args.out_dir / experiment_id}")
        print()
    return 0


def _report_p2p(
    args: argparse.Namespace,
    cfg: ScenarioConfig,
    tariffs: TariffSchedule,
    tariff_text: str,
    profiles: pd.DataFrame,
    results: Sequence[MarketResult],
    mechanism: str,
) -> None:
    trades = results_to_trade_frame(results)
    summary = summarize_settlements(results)
    baseline_total = math.fsum(r.total_baseline_cost_inr for r in results)
    mechanism_total = math.fsum(r.total_cost_inr for r in results)
    gain = math.fsum(r.total_gain_inr for r in results)
    volume = math.fsum(r.total_p2p_volume_kwh for r in results)
    fees = math.fsum(r.total_fee_revenue_inr for r in results)
    trading_hours = sum(1 for r in results if r.trades)

    print(
        f"Profiles: {args.profiles} ({len(profiles)} validated rows); {tariff_text}; "
        f"fees: seller INR {cfg.seller_fee_inr_per_kwh:.2f}/kWh, buyer INR {cfg.buyer_fee_inr_per_kwh:.2f}/kWh"
    )
    print("Signed costs: positive = net payment, negative = net income.")
    print()
    if trades.empty:
        print("No P2P trades were settled in any interval.")
    elif len(trades) <= MAX_PRINTED_TRADES:
        print(f"{'day':>3} {'hour':>4}  {'seller':<8} {'buyer':<8} {'kWh':>10} {'price INR/kWh':>14}")
        for row in trades.itertuples(index=False):
            print(
                f"{row.day:>3} {row.hour:>4}  {row.seller_id:<8} {row.buyer_id:<8} "
                f"{row.quantity_kwh:>10.3f} {row.price_inr_per_kwh:>14.4f}"
            )
    else:
        print(f"{len(trades)} trades in {trading_hours} interval(s); use --out-dir to write the full trade log.")
    print()
    print(
        f"{'microgrid':<10} {'P2P sold kWh':>13} {'P2P bought kWh':>15} "
        f"{'baseline INR':>13} {mechanism + ' INR':>16} {'gain INR':>10}"
    )
    for row in summary.itertuples(index=False):
        print(
            f"{row.microgrid_id:<10} {row.p2p_export_kwh:>13.3f} {row.p2p_import_kwh:>15.3f} "
            f"{row.baseline_cost_inr:>13.2f} {row.p2p_cost_inr:>16.2f} {row.gain_inr:>10.2f}"
        )
    print()
    print(f"Community baseline net cost: INR {baseline_total:.2f}")
    print(f"Community {mechanism} net cost: INR {mechanism_total:.2f}")
    print(f"Community gain: INR {gain:.2f} (P2P volume {volume:.3f} kWh; operator fees INR {fees:.2f})")
    fairness = jain_fairness(summary.gain_inr, tol=cfg.tol)
    print(
        "Jain fairness of per-player gains: "
        + ("N/A (no gains)" if fairness is None else f"{fairness:.8f}")
    )
    _write_outputs(args, cfg, tariffs, results, mechanism)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands = {
        "generate": _cmd_generate,
        "baseline": _cmd_baseline,
        "cooperative": _cmd_cooperative,
        "competitive": _cmd_competitive,
        "compare": _cmd_compare,
        "experiment": _cmd_experiment,
    }
    try:
        return commands[args.command](args)
    except (DataValidationError, EnergyBalanceError, SettlementError, SolverError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
