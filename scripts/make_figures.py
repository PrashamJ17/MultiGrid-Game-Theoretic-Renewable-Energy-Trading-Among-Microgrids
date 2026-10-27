"""Report figures from saved experiment results only (blueprint section 13, "Minimum report figures").

Usage (from the repository root, after `python -m microgrid.cli experiment all --out-dir results/experiments`):

    python scripts/make_figures.py --results results/experiments --out docs/figures

Every figure is drawn from the CSV files written by the experiment runner; nothing is recomputed here. The
script lives outside the ``microgrid`` package because the package is not allowed to import plotting libraries.

Conventions (validated palette, one y-axis per panel, units in every axis label, the mechanism and horizon in
every title, identity never by colour alone): microgrids use categorical slots in fixed ID order; mechanisms
use grey (baseline), violet (cooperative) and red (competitive); competitive interval states use the reserved
status colours with text labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MECHANISM = {"baseline": "#898781", "cooperative": "#6250d6", "competitive": "#e34948"}
LABEL = {"baseline": "Baseline (utility only)", "cooperative": "Cooperative (LP + Nash bargaining)",
         "competitive": "Competitive (seller price game)"}
STATUS = {"pure NE": "#0ca30c", "no pure NE": "#fab219", "cap exceeded": "#ec835a"}
INK, SECONDARY, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": SECONDARY,
        "axes.titlecolor": INK,
        "xtick.color": SECONDARY,
        "ytick.color": SECONDARY,
        "font.size": 9,
        "axes.titlesize": 10,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "lines.markersize": 6,
    }
)


def _axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save(fig, out: Path, name: str, title: str) -> Path:
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, color=INK, fontweight="bold")
    fig.tight_layout()
    path = out / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def _legend_below(ax, ncol: int = 1) -> None:
    """Legend under the plot area so it never covers data."""
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=ncol, fontsize=8)


def _colors(ids) -> dict[str, str]:
    return {mg: SERIES[i] for i, mg in enumerate(sorted(ids))}


def _mechanism_lines(ax, runs: pd.DataFrame, x: str, y: str, mechanisms=("cooperative", "competitive")) -> None:
    for name in mechanisms:
        part = runs[runs.mechanism == name]
        ax.plot(part[x], part[y], marker="o", color=MECHANISM[name], label=LABEL[name])


# --- the eight minimum figures -------------------------------------------------


def fig01_profiles(e1: Path, out: Path) -> Path:
    profiles = pd.read_csv(e1 / "profiles.csv")
    ids = sorted(profiles.microgrid_id.unique())
    colors = _colors(ids)
    fig, axes = plt.subplots(2, (len(ids) + 1) // 2, figsize=(10, 6), sharex=True, sharey=True)
    for ax, mg in zip(axes.ravel(), ids):
        part = profiles[profiles.microgrid_id == mg]
        ax.plot(part.hour, part.generation_kwh, color=colors[mg], label="Generation")
        ax.plot(part.hour, part.demand_kwh, color=SECONDARY, linewidth=1.5, label="Demand")
        _axes(ax, mg, "Hour of day", "kWh per hour")
        ax.legend(loc="upper left")
    return _save(fig, out, "fig01_profiles", "Fig. 1  Synthetic 24-hour generation and demand per microgrid (seed 42)")


def fig02_surplus_deficit(e1: Path, out: Path) -> Path:
    p = pd.read_csv(e1 / "profiles.csv")
    p["surplus"] = (p.generation_kwh - p.demand_kwh).clip(lower=0)
    p["deficit"] = (p.demand_kwh - p.generation_kwh).clip(lower=0)
    hourly = p.groupby("hour")[["surplus", "deficit"]].sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(hourly.index, hourly.surplus, color=SECONDARY, width=0.7, label="Total surplus (up)")
    ax.bar(hourly.index, -hourly.deficit, color=MUTED, width=0.7, label="Total deficit (down)")
    ax.axhline(0, color=AXIS, linewidth=0.8)
    _axes(ax, "Community totals, 24 h synthetic day", "Hour of day", "kWh per hour")
    ax.legend(loc="lower left")
    return _save(fig, out, "fig02_surplus_deficit", "Fig. 2  Hourly total surplus and demand deficit")


def fig03_pairs(e1: Path, out: Path) -> Path:
    trades = pd.read_csv(e1 / "trades.csv")
    trades["pair"] = trades.seller_id + "→" + trades.buyer_id
    pairs = sorted(trades.pair.unique())
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.38
    for k, name in enumerate(("cooperative", "competitive")):
        totals = trades[trades.mechanism == name].groupby("pair").quantity_kwh.sum().reindex(pairs, fill_value=0.0)
        xs = [i + (k - 0.5) * width for i in range(len(pairs))]
        bars = left.bar(xs, totals.values, width=width * 0.92, color=MECHANISM[name], label=LABEL[name])
        left.bar_label(bars, fmt="%.0f", padding=2, color=SECONDARY, fontsize=8)
        hourly = trades[trades.mechanism == name].groupby("hour").quantity_kwh.sum().reindex(range(24), fill_value=0)
        right.plot(hourly.index, hourly.values, marker="o", color=MECHANISM[name], label=LABEL[name])
    left.set_xticks(range(len(pairs)), pairs)
    _axes(left, "Energy per seller→buyer pair, whole day", "Matched pair", "kWh")
    _axes(right, "Hourly P2P volume", "Hour of day", "kWh per hour")
    left.legend(loc="upper right")
    return _save(fig, out, "fig03_p2p_pairs", "Fig. 3  P2P quantities and matched pairs by mechanism (24 h)")


def fig04_net_cost(e1: Path, out: Path) -> Path:
    runs = pd.read_csv(e1 / "runs.csv").set_index("mechanism")
    fig, ax = plt.subplots(figsize=(7, 4))
    names = ["baseline", "cooperative", "competitive"]
    bars = ax.bar([LABEL[n] for n in names], [runs.loc[n, "net_community_cost_inr"] for n in names],
                  color=[MECHANISM[n] for n in names], width=0.6)
    ax.bar_label(bars, labels=[f"{runs.loc[n, 'net_community_cost_inr']:,.0f}" for n in names], padding=3,
                 color=SECONDARY)
    _axes(ax, "Whole 24 h day, signed (positive = community pays)", "Mechanism", "INR")
    return _save(fig, out, "fig04_net_cost", "Fig. 4  Community net cost by mechanism")


def fig05_gains(e1: Path, out: Path) -> Path:
    players = pd.read_csv(e1 / "players.csv")
    runs = pd.read_csv(e1 / "runs.csv").set_index("mechanism")
    ids = sorted(players.microgrid_id.unique())
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.38
    for k, name in enumerate(("cooperative", "competitive")):
        part = players[players.mechanism == name].set_index("microgrid_id").reindex(ids)
        xs = [i + (k - 0.5) * width for i in range(len(ids))]
        jain = runs.loc[name, "jain_fairness"]
        bars = ax.bar(xs, part.gain_inr.values, width=width * 0.92, color=MECHANISM[name],
                      label=f"{LABEL[name]}, Jain fairness {jain:.3f}")
        ax.bar_label(bars, fmt="%.0f", padding=2, color=SECONDARY, fontsize=8)
    ax.set_xticks(range(len(ids)), ids)
    _axes(ax, "Gain vs baseline over the 24 h day", "Microgrid", "INR")
    ax.legend(loc="upper right")
    return _save(fig, out, "fig05_player_gains", "Fig. 5  Per-microgrid gains and fairness")


def fig06_seller_utility(e1: Path, out: Path) -> Path:
    curves = pd.read_csv(e1 / "payoff_curves.csv")
    two = curves.groupby("hour").seller.nunique()
    nash_hours = curves[curves.held_profile_is_nash].hour.unique()
    chosen_nash = next(h for h in nash_hours if two[h] == 2)
    chosen_none = curves[~curves.held_profile_is_nash].hour.min()
    colors = _colors(curves.seller.unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, hour in zip(axes, (chosen_nash, chosen_none)):
        part = curves[curves.hour == hour]
        held = ", ".join(f"{s}={p:.2f}" for s, p in part.groupby("seller").held_price_inr_per_kwh.first().items())
        for seller, group in part.groupby("seller"):
            ax.plot(group.own_price_inr_per_kwh, group.utility_inr, marker="o", color=colors[seller], label=seller)
            at = group[group.own_price_inr_per_kwh == group.held_price_inr_per_kwh.iloc[0]]
            ax.plot(at.own_price_inr_per_kwh, at.utility_inr, marker="o", markersize=12, color=colors[seller],
                    markeredgecolor=INK, linestyle="none")
        state = "verified pure NE" if bool(part.held_profile_is_nash.iloc[0]) else "NO PURE NE: held at NOT NASH diagnostic"
        _axes(ax, f"Hour {hour}: {state}\nothers held at {held}", "Own posted price (INR/kWh)",
              "Utility vs exporting (INR)")
        ax.legend(loc="upper left")
    return _save(fig, out, "fig06_seller_utility",
                 "Fig. 6  Competitive seller utility as one seller changes price (large marker = held price)")


def fig07_fees(e3: Path, out: Path) -> Path:
    runs = pd.read_csv(e3 / "runs.csv")
    x = "total_fee_inr_per_kwh"
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    _mechanism_lines(left, runs, x, "total_p2p_volume_kwh")
    _mechanism_lines(right, runs, x, "total_gain_inr")
    for ax in (left, right):
        ax.axvline(5.0, color=AXIS, linewidth=1, label="Fees = spread b − a (5 INR/kWh)")
    _axes(left, "P2P volume, 24 h", "Seller + buyer fee (INR/kWh)", "kWh")
    _axes(right, "Community gain vs baseline, 24 h", "Seller + buyer fee (INR/kWh)", "INR")
    _legend_below(left, 2)
    _legend_below(right, 2)
    return _save(fig, out, "fig07_fee_sensitivity", "Fig. 7  Sensitivity of trade volume and welfare to fees (E3)")


def fig08_market_size(e5: Path, out: Path) -> Path:
    runs = pd.read_csv(e5 / "runs.csv")
    runtimes = pd.read_csv(e5 / "runtimes.csv")
    comp = runs[runs.mechanism == "competitive"].set_index("microgrids")
    sizes = list(comp.index)
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(12, 4))
    bottom = [0] * len(sizes)
    for label, column in (("pure NE", "pure_ne_intervals"), ("no pure NE", "no_pure_ne_intervals"),
                          ("cap exceeded", "cap_exceeded_intervals")):
        values = comp[column].tolist()
        a.bar([str(s) for s in sizes], values, bottom=bottom, color=STATUS[label], label=label, width=0.6)
        bottom = [x + y for x, y in zip(bottom, values)]
    _axes(a, "Competitive intervals by status", "Microgrids", "Intervals with a game")
    _legend_below(a, 3)
    regret = comp["max_not_nash_regret_inr"].fillna(0.0)
    b.bar([str(s) for s in sizes], regret, color=MECHANISM["competitive"], width=0.6)
    labels = []
    for size in sizes:
        row = comp.loc[size]
        text = f"{row.max_not_nash_regret_inr:.1f}" if row.no_pure_ne_intervals else "none"
        if row.cap_exceeded_intervals:
            text += f"\n({int(row.cap_exceeded_intervals)} h capped)"
        labels.append(text)
    b.bar_label(b.containers[0], labels=labels, padding=2, color=SECONDARY, fontsize=8)
    b.set_ylim(0, max(regret.max(), 1.0) * 1.3)
    _axes(b, "Worst NOT NASH regret (settled NE regret = 0)", "Microgrids", "INR")
    for name in ("baseline", "cooperative", "competitive"):
        part = runtimes[runtimes.mechanism == name]
        c.plot(part.microgrids, part.runtime_seconds * 1000, marker="o", color=MECHANISM[name], label=LABEL[name])
    c.set_xticks(sizes)
    _axes(c, "Solve time for the 24 h day", "Microgrids", "milliseconds")
    _legend_below(c)
    return _save(fig, out, "fig08_market_size", "Fig. 8  Equilibrium regret and runtime by market size (E5)")


# --- one figure per remaining experiment -----------------------------------------


def fig09_spread(e2: Path, out: Path) -> Path:
    runs = pd.read_csv(e2 / "runs.csv")
    x = "spread_inr_per_kwh"
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    _mechanism_lines(left, runs, x, "total_gain_inr")
    comp = runs[runs.mechanism == "competitive"]
    right.bar(comp[x], comp.no_pure_ne_intervals, color=STATUS["no pure NE"], width=0.6)
    _axes(left, "Community gain vs baseline, 24 h", "Spread b − a (INR/kWh)", "INR")
    _axes(right, "Competitive intervals with no pure NE", "Spread b − a (INR/kWh)", "Intervals")
    _legend_below(left, 2)
    return _save(fig, out, "fig09_spread", "Fig. 9  Welfare vs retail-export spread (E2, export price 3 INR/kWh)")


def fig10_solar(e4: Path, out: Path) -> Path:
    runs = pd.read_csv(e4 / "runs.csv")
    x = "solar_scale"
    base = runs[runs.mechanism == "baseline"]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(12, 4))
    a.plot(base[x], base.total_surplus_kwh, marker="o", color=SECONDARY, label="Total surplus")
    a.plot(base[x], base.total_deficit_kwh, marker="s", color=MUTED, label="Total deficit")
    _mechanism_lines(b, runs, x, "total_p2p_volume_kwh")
    _mechanism_lines(c, runs, x, "net_community_cost_inr", ("baseline", "cooperative", "competitive"))
    _axes(a, "Energy position, 24 h", "Solar scale (x)", "kWh")
    _axes(b, "P2P volume, 24 h", "Solar scale (x)", "kWh")
    _axes(c, "Community net cost, 24 h", "Solar scale (x)", "INR")
    for ax in (a, b, c):
        ax.set_xticks(sorted(runs[x].unique()))
        _legend_below(ax)
    return _save(fig, out, "fig10_solar_scale", "Fig. 10  Surplus, traded volume and cost vs solar scale (E4)")


def fig11_tick(e6: Path, out: Path) -> Path:
    runs = pd.read_csv(e6 / "runs.csv")
    runtimes = pd.read_csv(e6 / "runtimes.csv")
    x = "price_step_inr"
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(12, 4))
    _mechanism_lines(a, runs, x, "average_price_inr_per_kwh")
    _mechanism_lines(b, runs, x, "jain_fairness")
    comp = runtimes[runtimes.mechanism == "competitive"]
    c.plot(comp[x], comp.runtime_seconds * 1000, marker="o", color=MECHANISM["competitive"])
    _axes(a, "Volume-weighted average P2P price", "Price tick (INR/kWh)", "INR/kWh")
    _axes(b, "Jain fairness of per-microgrid gains", "Price tick (INR/kWh)", "Index (1 = equal)")
    _axes(c, "Competitive solve time, 24 h", "Price tick (INR/kWh)", "milliseconds")
    for ax in (a, b, c):
        ax.set_xticks(sorted(runs[x].unique()))
        ax.invert_xaxis()  # finer ticks to the right
    _legend_below(a)
    _legend_below(b)
    return _save(fig, out, "fig11_price_tick", "Fig. 11  Strategy resolution: price tick 1.0 / 0.5 / 0.25 (E6)")


def fig12_seeds(e7: Path, out: Path) -> Path:
    runs = pd.read_csv(e7 / "runs.csv")
    x = "seed"
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, ylabel in ((left, "total_gain_inr", "INR"), (right, "jain_fairness", "Index (1 = equal)")):
        for name in ("cooperative", "competitive"):
            part = runs[runs.mechanism == name]
            ax.plot(part[x], part[metric], marker="o", linestyle="none", color=MECHANISM[name], label=LABEL[name])
            ax.axhline(part[metric].mean(), color=MECHANISM[name], linewidth=1)
        ax.set_xticks(sorted(runs[x].unique()))
        _legend_below(ax, 2)
        _axes(ax, ("Community gain" if metric == "total_gain_inr" else "Jain fairness")
              + " per seed (line = mean)", "Seed", ylabel)
    return _save(fig, out, "fig12_seeds", "Fig. 12  Variation across ten synthetic days, seeds 42–51 (E7)")


def make_all(results: Path, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    return [
        fig01_profiles(results / "E1", out),
        fig02_surplus_deficit(results / "E1", out),
        fig03_pairs(results / "E1", out),
        fig04_net_cost(results / "E1", out),
        fig05_gains(results / "E1", out),
        fig06_seller_utility(results / "E1", out),
        fig07_fees(results / "E3", out),
        fig08_market_size(results / "E5", out),
        fig09_spread(results / "E2", out),
        fig10_solar(results / "E4", out),
        fig11_tick(results / "E6", out),
        fig12_seeds(results / "E7", out),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path("results/experiments"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    for path in make_all(args.results, args.out):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
