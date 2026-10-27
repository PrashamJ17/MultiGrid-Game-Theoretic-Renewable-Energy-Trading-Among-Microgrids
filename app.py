"""MultiGrid Streamlit dashboard (Phase 4).

A thin adapter: every number comes from the ``microgrid`` package
(``simulation.compare_all`` for all three mechanisms,
``competitive.analyze_price_game`` for the game analysis). This file only
gathers inputs, lays out the pages and draws charts. Energy profiles depend
only on the scenario controls, so changing tariffs, fees or mechanism
parameters never regenerates them.

Run from the repository root:
    python -m streamlit run app.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    # Lets `streamlit run app.py` work even without `pip install -e .`.
    sys.path.insert(0, str(_SRC))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from microgrid import __version__  # noqa: E402
from microgrid.competitive import analyze_price_game  # noqa: E402
from microgrid.data import diagnostics_records, generate_profiles, load_and_validate_profiles, load_config  # noqa: E402
from microgrid.energy import add_surplus_deficit_columns, compute_hour_states  # noqa: E402
from microgrid.models import (  # noqa: E402
    CombinatorialWarning,
    DataValidationError,
    EconomicWarning,
    EnergyBalanceError,
    ScenarioConfig,
    SettlementError,
    SolverError,
)
from microgrid.simulation import ComparisonResult, compare_all  # noqa: E402

GOLDEN = "Golden one-hour fixture (blueprint section 4)"
SAMPLE = "Synthetic 24 h, seed 42 (committed CSV)"
CUSTOM = "Synthetic 24 h, custom seed and solar scale"
SCENARIOS = (GOLDEN, SAMPLE, CUSTOM)

MECHANISMS = ("baseline", "cooperative", "competitive")
LABELS = {
    "baseline": "Baseline (utility only)",
    "cooperative": "Cooperative (LP + Nash bargaining)",
    "competitive": "Competitive (seller price game)",
}

# Validated with the dataviz palette validator (categorical slots in fixed order; light and dark steps).
PALETTE = {
    "light": {
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
        "mechanism": {"baseline": "#898781", "cooperative": "#6250d6", "competitive": "#e34948"},
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "sequential": ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
    },
    "dark": {
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
        "mechanism": {"baseline": "#898781", "cooperative": "#9085e9", "competitive": "#e66767"},
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "sequential": ["#0d366b", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"],
    },
}
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


# ---------------------------------------------------------------------------
# Cached package calls
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_scenario(scenario: str, seed: int, solar_scale: float, microgrids: int) -> tuple[pd.DataFrame, dict]:
    """Validated profiles and the base config of a scenario (independent of tariffs and fees)."""
    if scenario == GOLDEN:
        cfg = load_config(ROOT / "configs" / "golden_one_hour.json")
        profiles = load_and_validate_profiles(ROOT / "data" / "golden_one_hour.csv", cfg)
    elif scenario == SAMPLE:
        cfg = load_config(ROOT / "configs" / "default.json")
        profiles = load_and_validate_profiles(ROOT / "data" / "sample_24h.csv", cfg)
    else:
        base = load_config(ROOT / "configs" / "default.json")
        cfg = dataclasses.replace(
            base,
            scenario_id=f"synthetic_seed{seed}_solar{solar_scale:g}_{microgrids}mg",
            seed=seed,
            microgrid_ids=tuple(f"MG{i}" for i in range(1, microgrids + 1)),
            generator=dataclasses.replace(base.generator, solar_scale=solar_scale),
        )
        profiles = generate_profiles(cfg)
    return profiles, cfg.to_dict()


@st.cache_data(show_spinner="Running all three mechanisms ...")
def run_all(profiles: pd.DataFrame, cfg_json: str) -> tuple[ComparisonResult, list[str]]:
    """All mechanisms on the same frozen profiles, plus economic warnings raised by the config.

    Per-interval enumeration-cap warnings are summarised by ``equilibrium_banner`` instead.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", CombinatorialWarning)
        cfg = ScenarioConfig.from_dict(json.loads(cfg_json))
        comparison = compare_all(profiles, cfg)
    notes = sorted({str(w.message) for w in caught if issubclass(w.category, EconomicWarning)})
    return comparison, notes


@st.cache_data(show_spinner=False)
def game_analysis(profiles: pd.DataFrame, cfg_json: str, day: int, hour: int):
    cfg = ScenarioConfig.from_dict(json.loads(cfg_json))
    state = next(s for s in compute_hour_states(profiles, cfg.tol) if (s.day, s.hour) == (day, hour))
    return analyze_price_game(state, cfg)


# ---------------------------------------------------------------------------
# Presentation helpers (no market maths)
# ---------------------------------------------------------------------------


def theme_name() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:  # older Streamlit or bare test runs
        return "light"


def style(fig: go.Figure, pal: dict, title: str, x_title: str, y_title: str, *, hover: str = "closest",
          height: int = 380, right: int = 16) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        height=height,
        margin=dict(l=8, r=right, t=72, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title_text=""),
        hovermode=hover,
        bargap=0.3,
        bargroupgap=0.08,
        barcornerradius=4,
        font=dict(family=FONT),
    )
    fig.update_xaxes(title_text=x_title, showgrid=False, linecolor=pal["axis"], ticks="outside", tickcolor=pal["axis"])
    fig.update_yaxes(title_text=y_title, gridcolor=pal["grid"], zeroline=True, zerolinecolor=pal["axis"])
    return fig


def end_labels(fig: go.Figure, pal: dict, points: dict[str, tuple[float, float]]) -> None:
    """Selective direct labels at line ends (at most four series); text uses ink, not series colour.

    A series is labelled only if its end point is clear of every other end point (8% of the y-range), so
    labels never collide; colliding series rely on the legend, which is always present.
    """
    if len(points) > 4:
        return
    values = [float(v) for trace in fig.data for v in (trace.y if trace.y is not None else [])]
    gap = 0.08 * (max(values) - min(values)) if values and max(values) > min(values) else 1.0
    for name, (x, y) in points.items():
        if any(abs(y - other_y) < gap for other, (_, other_y) in points.items() if other != name):
            continue
        fig.add_annotation(x=x, y=y, text=name, showarrow=False, xanchor="left", xshift=6,
                           font=dict(color=pal["secondary"], size=12))


def show(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, theme="streamlit", key=key)


def table_view(frame: pd.DataFrame, label: str = "Table view") -> None:
    with st.expander(label):
        st.dataframe(frame, hide_index=True)


def fmt(value, template: str = "{:,.2f}") -> str:
    return "N/A" if value is None else template.format(value)


def profile_text(pairs) -> str:
    return ", ".join(f"{s}={p:.2f}" for s, p in pairs) if pairs else ""


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def sidebar_inputs() -> tuple[pd.DataFrame, dict]:
    sb = st.sidebar
    sb.header("Scenario")
    scenario = sb.selectbox("Energy profiles", SCENARIOS, key="scenario")
    seed, solar_scale, microgrids = 42, 1.0, 4
    if scenario == CUSTOM:
        seed = int(sb.number_input("Random seed", min_value=0, max_value=100_000, value=42, step=1, key="seed"))
        solar_scale = float(sb.slider("Solar scale (x)", 0.5, 2.0, 1.0, 0.25, key="solar_scale"))
        microgrids = int(sb.selectbox("Microgrids", (4, 6), key="microgrids"))
    profiles, base = load_scenario(scenario, seed, solar_scale, microgrids)
    sb.caption("Profiles are fixed by the scenario; the controls below never regenerate them.")

    sb.header("Tariffs and fees (INR/kWh)")
    retail = sb.slider("Retail import price b", 4.0, 12.0, float(base["retail_price_inr_per_kwh"]), 0.5, key="retail")
    export = sb.slider("Export price a", 0.0, 6.0, float(base["export_price_inr_per_kwh"]), 0.5, key="export")
    seller_fee = sb.slider("Seller fee f_s", 0.0, 5.0, float(base["seller_fee_inr_per_kwh"]), 0.25, key="seller_fee")
    buyer_fee = sb.slider("Buyer fee f_b", 0.0, 5.0, float(base["buyer_fee_inr_per_kwh"]), 0.25, key="buyer_fee")

    sb.header("Mechanism parameters")
    alpha = sb.slider("Seller bargaining weight alpha (cooperative)", 0.05, 0.95,
                      float(base["bargaining_seller_weight"]), 0.05, key="alpha")
    tick = sb.selectbox("Price tick for the seller game (INR/kWh)", (1.0, 0.5, 0.25), index=1, key="price_step")
    cfg = {
        **base,
        "retail_price_inr_per_kwh": round(retail, 4),
        "export_price_inr_per_kwh": round(export, 4),
        "seller_fee_inr_per_kwh": round(seller_fee, 4),
        "buyer_fee_inr_per_kwh": round(buyer_fee, 4),
        "bargaining_seller_weight": round(alpha, 4),
        "price_step_inr": float(tick),
    }
    return profiles, cfg


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_overview(comparison: ComparisonResult, profiles: pd.DataFrame) -> None:
    cfg = comparison.config
    m = comparison.metrics
    st.markdown(
        "Grid-connected microgrids with rooftop solar either have **surplus** energy (generation above demand) "
        "or a **deficit** in each hour. Without P2P trading they export surplus to the utility at the export "
        "price *a* and import deficits at the retail price *b*. Trading locally can capture part of the spread "
        "*b − a*. This dashboard compares three ways of organising that trade on the **same** energy profiles:"
    )
    st.markdown(
        "- **Baseline:** utility only, no P2P trading.\n"
        "- **Cooperative:** a welfare-maximising LP fixes who trades how much; each pair then splits its trade "
        "surplus by bilateral Nash bargaining.\n"
        "- **Competitive:** sellers post prices simultaneously; buyers take the cheapest acceptable offer. Only "
        "outcomes verified as pure Nash equilibria are settled."
    )
    intervals = profiles[["day", "hour"]].drop_duplicates().shape[0]
    cols = st.columns(4)
    cols[0].metric("Microgrids", len(cfg.microgrid_ids))
    cols[1].metric("Hourly intervals", intervals)
    cols[2].metric("Tariff spread b − a (INR/kWh)", f"{cfg.retail_price_inr_per_kwh - cfg.export_price_inr_per_kwh:.2f}")
    cols[3].metric("Fees f_s + f_b (INR/kWh)", f"{cfg.seller_fee_inr_per_kwh + cfg.buyer_fee_inr_per_kwh:.2f}")

    st.subheader("Community net cost over the horizon (INR)")
    cols = st.columns(3)
    baseline_cost = m["baseline"]["net_community_cost_inr"]
    for col, name in zip(cols, MECHANISMS):
        cost = m[name]["net_community_cost_inr"]
        col.metric(
            LABELS[name],
            f"{cost:,.2f}",
            delta=None if name == "baseline" else f"{cost - baseline_cost:,.2f} vs baseline",
            delta_color="inverse",
        )
    st.caption("Signed costs: positive = the community pays the utility; negative = net export revenue.")
    equilibrium_banner(comparison)


def equilibrium_banner(comparison: ComparisonResult) -> None:
    comp = comparison.metrics["competitive"]
    if comp["no_pure_ne_intervals"]:
        st.error(
            f"NO PURE EQUILIBRIUM FOUND in {comp['no_pure_ne_intervals']} interval(s) of the competitive game. "
            "Those intervals are settled without P2P trades; their diagnostic price profiles are NOT NASH."
        )
    if comp["cap_exceeded_intervals"]:
        st.warning(
            f"{comp['cap_exceeded_intervals']} interval(s) had more sellers than max_competitive_sellers; exact "
            "equilibrium enumeration was not attempted there and nothing is claimed or settled."
        )


def page_profiles(profiles: pd.DataFrame, cfg: ScenarioConfig, pal: dict, colors: dict[str, str]) -> None:
    frame = add_surplus_deficit_columns(profiles, tol=cfg.tol)
    frame["t"] = frame["day"] * 24 + frame["hour"]
    st.caption("All profiles are synthetic or blueprint fixtures, in kWh per one-hour interval.")
    if frame["t"].nunique() == 1:
        hour = int(frame["hour"].iloc[0])
        fig = go.Figure()
        for row in frame.itertuples(index=False):
            fig.add_bar(x=[row.microgrid_id], y=[row.net_kwh], name=row.microgrid_id, marker_color=colors[row.microgrid_id],
                        text=[f"{row.net_kwh:+.1f}"], textposition="outside",
                        hovertemplate="%{x}: %{y:+.2f} kWh<extra></extra>")
        style(fig, pal, f"Net position G − D per microgrid, hour {hour} (kWh): surplus up, deficit down",
              "Microgrid", "kWh", height=360)
        fig.update_layout(showlegend=False)
        show(fig, "profile_net_single")
    else:
        for column, title, key in (
            ("generation_kwh", "Generation per microgrid (kWh per hour), synthetic 24 h", "gen"),
            ("demand_kwh", "Demand per microgrid (kWh per hour), synthetic 24 h", "dem"),
            ("net_kwh", "Net position G − D per microgrid (kWh): above zero = surplus, below = deficit", "net"),
        ):
            fig = go.Figure()
            ends = {}
            for mg in sorted(cfg.microgrid_ids):
                part = frame[frame.microgrid_id == mg]
                fig.add_scatter(x=part["t"], y=part[column], name=mg, mode="lines", line=dict(width=2, color=colors[mg]),
                                hovertemplate=f"{mg}: %{{y:.2f}} kWh<extra></extra>")
                ends[mg] = (float(part["t"].iloc[-1]), float(part[column].iloc[-1]))
            style(fig, pal, title, "Hour of horizon", "kWh", hover="x unified", right=56)
            end_labels(fig, pal, ends)
            show(fig, f"profile_{key}")
        hourly = frame.groupby("t", as_index=False)[["surplus_kwh", "deficit_kwh"]].sum()
        fig = go.Figure()
        fig.add_bar(x=hourly["t"], y=hourly["surplus_kwh"], name="Total surplus (up)", marker_color=pal["secondary"],
                    hovertemplate="surplus %{y:.2f} kWh<extra></extra>")
        fig.add_bar(x=hourly["t"], y=-hourly["deficit_kwh"], name="Total deficit (down)", marker_color=pal["muted"],
                    hovertemplate="deficit %{y:.2f} kWh<extra></extra>")
        style(fig, pal, "Hourly community surplus (up) and deficit (down), kWh", "Hour of horizon", "kWh",
              hover="x unified")
        fig.update_layout(barmode="relative")
        show(fig, "profile_totals")
    table_view(frame.drop(columns=["t"]))


def page_simulate(comparison: ComparisonResult) -> None:
    cfg = comparison.config
    st.markdown("Change the controls in the sidebar; all three mechanisms re-run on the same frozen profiles.")
    params = pd.DataFrame(
        [
            ("Scenario", cfg.scenario_id),
            ("Retail import price b (INR/kWh)", f"{cfg.retail_price_inr_per_kwh:.2f}"),
            ("Export price a (INR/kWh)", f"{cfg.export_price_inr_per_kwh:.2f}"),
            ("Seller fee f_s (INR/kWh)", f"{cfg.seller_fee_inr_per_kwh:.2f}"),
            ("Buyer fee f_b (INR/kWh)", f"{cfg.buyer_fee_inr_per_kwh:.2f}"),
            ("Seller bargaining weight alpha", f"{cfg.bargaining_seller_weight:.2f}"),
            ("Seller-game price tick (INR/kWh)", f"{cfg.price_step_inr:.2f}"),
            ("Max sellers for exact enumeration", str(cfg.max_competitive_sellers)),
        ],
        columns=["Parameter", "Value"],
    )
    st.dataframe(params, hide_index=True)
    st.markdown(
        "- P2P trade is worthwhile only while **b − a − f_s − f_b > 0**; at or beyond that threshold no "
        "mechanism trades.\n"
        "- **alpha** only moves the cooperative price between the reservation prices; it redistributes gain "
        "without changing the total.\n"
        "- The **price tick** defines the seller game's strategy sets, so it can change which profiles are "
        "equilibria."
    )
    summary_table(comparison)
    runtimes = ", ".join(f"{name} {seconds * 1000:.1f} ms" for name, seconds in comparison.runtimes_seconds)
    env = ", ".join(f"{k} {v}" for k, v in comparison.environment if k != "platform")
    st.caption(f"Solve time: {runtimes}. Environment: {env}.")


SUMMARY_ROWS = (
    ("Community net cost (INR)", "net_community_cost_inr", "{:,.2f}"),
    ("Gain vs baseline (INR)", "total_gain_inr", "{:,.2f}"),
    ("Cost reduction (%)", "cost_reduction_pct", "{:.2f}"),
    ("P2P volume (kWh)", "total_p2p_volume_kwh", "{:,.3f}"),
    ("Average P2P price (INR/kWh)", "average_price_inr_per_kwh", "{:.4f}"),
    ("Operator fees (INR)", "operator_fee_revenue_inr", "{:,.2f}"),
    ("Jain fairness of gains", "jain_fairness", "{:.4f}"),
    ("Efficiency vs LP optimum", "efficiency", "{:.4f}"),
    ("Trading intervals", "trading_intervals", "{}"),
    ("Pure-NE intervals", "pure_ne_intervals", "{}"),
    ("NO_PURE_NE intervals", "no_pure_ne_intervals", "{}"),
    ("Max regret at settled NE (INR)", "max_regret_inr", "{:.6f}"),
)


def summary_table(comparison: ComparisonResult) -> None:
    rows = [
        {"Metric": label, **{LABELS[n]: fmt(comparison.metrics[n][key], template) for n in comparison.mechanisms}}
        for label, key, template in SUMMARY_ROWS
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("N/A = not applicable: no P2P volume, baseline cost ≤ 0, all gains zero, or a game-only metric.")


def page_transactions(comparison: ComparisonResult, pal: dict) -> None:
    mechanism = st.radio("Mechanism", ("cooperative", "competitive"), format_func=LABELS.get, horizontal=True,
                         key="tx_mechanism")
    trades = comparison.trades_frame()
    chosen = trades[trades.mechanism == mechanism].drop(columns=["scenario_id", "mechanism"])
    if chosen.empty:
        st.info(f"No P2P trades under the {mechanism} mechanism for these parameters.")
    else:
        st.dataframe(chosen.sort_values(["day", "hour", "seller_id", "buyer_id"]), hide_index=True)

    hourly = (
        trades.groupby(["mechanism", "day", "hour"], as_index=False)["quantity_kwh"].sum()
        if not trades.empty else pd.DataFrame(columns=["mechanism", "day", "hour", "quantity_kwh"])
    )
    intervals = comparison.validated_profiles[["day", "hour"]].drop_duplicates()
    if len(intervals) > 1:
        fig = go.Figure()
        for name in ("cooperative", "competitive"):
            series = intervals.merge(hourly[hourly.mechanism == name], on=["day", "hour"], how="left").fillna(
                {"quantity_kwh": 0.0})
            t = series["day"] * 24 + series["hour"]
            fig.add_scatter(x=t, y=series["quantity_kwh"], name=LABELS[name], mode="lines+markers",
                            line=dict(width=2, color=pal["mechanism"][name]), marker=dict(size=8),
                            hovertemplate=f"{name}: %{{y:.2f}} kWh<extra></extra>")
        style(fig, pal, "Hourly P2P volume by mechanism (kWh), synthetic 24 h", "Hour of horizon", "kWh",
              hover="x unified")
        show(fig, "tx_hourly")
        table_view(hourly)

    st.subheader("Per-interval settlement")
    settlement = comparison.settlement_frame()
    st.dataframe(
        settlement[settlement.mechanism == mechanism].drop(columns=["scenario_id"]).rename(
            columns={"p2p_cost_inr": "net_cost_inr (negative = revenue)"}),
        hide_index=True,
    )


def page_game(comparison: ComparisonResult, profiles: pd.DataFrame, cfg_json: str, pal: dict,
              colors: dict[str, str]) -> None:
    st.markdown(
        "- **Players:** the sellers of the interval; buyers are non-strategic followers.\n"
        "- **Strategies:** posted prices on the tick, from the export price (+ seller fee) up to just below retail "
        "(− buyer fee).\n"
        "- **Buyer rule:** buyers in ID order take the cheapest offers first, ties broken by seller ID.\n"
        "- **Equilibrium test:** every unilateral price deviation of every seller in every profile is evaluated; "
        "a profile is a pure NE when no seller can strictly gain."
    )
    equilibrium_banner(comparison)
    records = [r for r in diagnostics_records(comparison.results_for("competitive"))
               if r["examined_profiles"] or r["status"] == "ENUMERATION_CAP_EXCEEDED"]
    if not records:
        st.info("No interval has both sellers and buyers with an individually rational price, so there is no game.")
        return
    status = pd.DataFrame(
        [
            {
                "day": r["day"],
                "hour": r["hour"],
                "status": r["status"],
                "pure NE count": r["equilibrium_count"],
                "profiles checked": r["examined_profiles"],
                "selected NE prices": profile_text((r["selected_profile"] or {}).items()),
                "NOT NASH diagnostic": profile_text((r["not_nash_profile"] or {}).items()),
                "max regret (INR)": r["max_unilateral_regret_inr"],
            }
            for r in records
        ]
    )
    st.dataframe(status, hide_index=True)

    playable = [(r["day"], r["hour"]) for r in records if r["examined_profiles"]]
    if not playable:
        return
    day, hour = st.selectbox("Interval to analyse", playable, format_func=lambda dh: f"day {dh[0]}, hour {dh[1]}",
                             key="game_interval")
    analysis = game_analysis(profiles, cfg_json, day, hour)
    tol = float(json.loads(cfg_json)["tol"])
    if analysis.equilibria:
        held = analysis.selected_equilibrium
        held_label = "the selected equilibrium"
        st.success(
            f"Verified pure Nash equilibrium: {profile_text(zip(analysis.sellers, held))} "
            f"({len(analysis.equilibria)} pure NE among {analysis.examined_profiles} profiles; "
            f"max regret INR {analysis.maximum_unilateral_regret(held):.6f})."
        )
    else:
        held = analysis.min_max_regret_profile
        held_label = "the NOT NASH diagnostic profile"
        st.error(
            f"NO PURE EQUILIBRIUM FOUND among {analysis.examined_profiles} profiles. Diagnostic profile "
            f"{profile_text(zip(analysis.sellers, held))} is NOT NASH: a seller can gain up to "
            f"INR {analysis.maximum_unilateral_regret(held):.4f} by deviating."
        )

    fig = go.Figure()
    rows = []
    for i, seller in enumerate(analysis.sellers):
        grid = analysis.price_grids[i]
        utilities = []
        for price in grid:
            profile = list(held)
            profile[i] = price
            utilities.append(analysis.payoffs(profile)[seller])
            rows.append({"seller": seller, "own price (INR/kWh)": price, "utility (INR)": utilities[-1]})
        fig.add_scatter(x=list(grid), y=utilities, name=seller, mode="lines+markers",
                        line=dict(width=2, color=colors[seller]), marker=dict(size=8),
                        hovertemplate=f"{seller} posts %{{x:.2f}}: utility %{{y:.2f}} INR<extra></extra>")
        fig.add_scatter(x=[held[i]], y=[analysis.payoffs(held)[seller]], mode="markers", showlegend=False,
                        marker=dict(size=14, color=colors[seller], line=dict(width=2, color=pal["secondary"])),
                        hovertemplate=f"{seller} held at %{{x:.2f}}<extra></extra>")
    style(fig, pal, f"Seller utility as its own price changes, others held at {held_label} (day {day}, hour {hour})",
          "Own posted price (INR/kWh)", "Utility vs exporting (INR)")
    show(fig, "game_payoff")
    table_view(pd.DataFrame(rows), "Table view: utility curves")

    if len(analysis.sellers) == 2:
        x_grid, y_grid = analysis.price_grids
        fig = go.Figure(
            go.Heatmap(
                x=list(x_grid), y=list(y_grid), z=analysis.max_regret.T,  # rows: second seller's price
                colorscale=[[k / 4, c] for k, c in enumerate(pal["sequential"])],
                colorbar=dict(title="INR"), xgap=2, ygap=2,
                hovertemplate=f"{analysis.sellers[0]}=%{{x:.2f}}, {analysis.sellers[1]}=%{{y:.2f}}: "
                              "max regret %{z:.2f} INR<extra></extra>",
            )
        )
        for eq in analysis.equilibria:
            fig.add_annotation(x=eq[0], y=eq[1], text="NE", showarrow=False,
                               font=dict(color=pal["secondary"], size=12))
        style(fig, pal, "Maximum unilateral regret over every price profile (INR); NE = pure equilibrium",
              f"{analysis.sellers[0]} price (INR/kWh)", f"{analysis.sellers[1]} price (INR/kWh)", height=460)
        fig.update_yaxes(zeroline=False)
        show(fig, "game_regret")

    shape = analysis.max_regret.shape
    table = []
    for flat, regret in enumerate(analysis.max_regret.ravel()):
        index, remainder = [], flat
        for size in reversed(shape):
            index.append(remainder % size)
            remainder //= size
        profile = [analysis.price_grids[i][k] for i, k in enumerate(reversed(index))]
        payoffs = analysis.payoffs(profile)
        table.append({**{f"{s} price": p for s, p in zip(analysis.sellers, profile)},
                      **{f"{s} utility (INR)": payoffs[s] for s in analysis.sellers},
                      "max regret (INR)": float(regret), "pure NE": bool(regret <= tol)})
    table_view(pd.DataFrame(table), "Table view: full payoff table")


def page_comparison(comparison: ComparisonResult, pal: dict) -> None:
    summary_table(comparison)
    m = comparison.metrics

    fig = go.Figure()
    for name in comparison.mechanisms:
        cost = m[name]["net_community_cost_inr"]
        fig.add_bar(x=[LABELS[name]], y=[cost], name=LABELS[name], marker_color=pal["mechanism"][name],
                    text=[f"{cost:,.2f}"], textposition="outside", hovertemplate="%{x}: INR %{y:,.2f}<extra></extra>")
    style(fig, pal, "Community net cost by mechanism (INR, whole horizon)", "Mechanism", "INR")
    fig.update_layout(showlegend=False)
    show(fig, "cmp_cost")

    players = comparison.player_frame()
    fig = go.Figure()
    for name in ("cooperative", "competitive"):
        part = players[players.mechanism == name]
        fig.add_bar(x=part["microgrid_id"], y=part["gain_inr"], name=LABELS[name], marker_color=pal["mechanism"][name],
                    hovertemplate=f"{name} %{{x}}: INR %{{y:,.2f}}<extra></extra>")
    style(fig, pal, "Gain vs baseline per microgrid (INR, whole horizon)", "Microgrid", "INR")
    fig.update_layout(barmode="group")
    show(fig, "cmp_players")
    table_view(players)

    hourly = pd.DataFrame(
        [
            {"mechanism": name, "t": r.day * 24 + r.hour, "gain_inr": r.total_gain_inr,
             "net_cost_inr": r.total_cost_inr, "status": r.diagnostics.status}
            for name, results in comparison.results
            for r in results
        ]
    )
    if hourly["t"].nunique() > 1:
        fig = go.Figure()
        for name in ("cooperative", "competitive"):
            part = hourly[hourly.mechanism == name]
            fig.add_scatter(x=part["t"], y=part["gain_inr"], name=LABELS[name], mode="lines+markers",
                            line=dict(width=2, color=pal["mechanism"][name]), marker=dict(size=8),
                            hovertemplate=f"{name}: INR %{{y:.2f}}<extra></extra>")
        style(fig, pal, "Hourly community gain vs baseline by mechanism (INR)", "Hour of horizon", "INR",
              hover="x unified")
        show(fig, "cmp_hourly")
        st.caption(
            "Where the competitive equilibrium trades the same volume as the cooperative LP the two lines coincide "
            "(the competitive line is drawn on top); gaps are intervals with a different volume or no pure equilibrium."
        )
        table_view(hourly)

    st.markdown(
        "**Reading the comparison**\n"
        "- Price renegotiation redistributes the P2P gain without changing the total at fixed quantities and fees.\n"
        "- When the competitive equilibrium moves the same quantity as the cooperative LP, total gains coincide; "
        "compare the distribution (per-player gains, Jain fairness) instead.\n"
        "- Cost reduction % is shown only when the baseline cost is positive; otherwise read the absolute gain.\n"
        "- Intervals without a pure equilibrium are settled without P2P trades in the competitive totals.\n"
        "- Higher P2P volume is not a claim about more renewable generation or lower emissions."
    )


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def page_network(comparison: ComparisonResult, pal: dict, colors: dict[str, str]) -> None:
    mechanism = st.radio("Mechanism", ("cooperative", "competitive"), format_func=LABELS.get, horizontal=True,
                         key="net_mechanism")
    trades = comparison.trades_frame()
    trades = trades[trades.mechanism == mechanism]
    if trades.empty:
        st.info("No P2P trades to draw for these parameters.")
        return
    flows = trades.groupby(["seller_id", "buyer_id"], as_index=False)["quantity_kwh"].sum()
    sellers = sorted(flows.seller_id.unique())
    buyers = sorted(flows.buyer_id.unique())
    fig = go.Figure(
        go.Sankey(
            node=dict(label=[f"{s} (sells)" for s in sellers] + [f"{b} (buys)" for b in buyers],
                      color=[colors[s] for s in sellers] + [colors[b] for b in buyers], pad=18, thickness=14),
            link=dict(
                source=[sellers.index(s) for s in flows.seller_id],
                target=[len(sellers) + buyers.index(b) for b in flows.buyer_id],
                value=flows.quantity_kwh.tolist(),
                color=[_rgba(colors[s], 0.35) for s in flows.seller_id],
                hovertemplate="%{source.label} → %{target.label}: %{value:.2f} kWh<extra></extra>",
            ),
        )
    )
    style(fig, pal, f"Contractual P2P energy, {LABELS[mechanism]} (kWh, whole horizon)", "", "", height=420)
    show(fig, "network_sankey")
    st.caption("Contracted quantities only; the simulator does not compute physical power flows.")
    table_view(flows)


def page_downloads(comparison: ComparisonResult) -> None:
    cfg = comparison.config
    stem = cfg.scenario_id
    st.markdown("Every file below reproduces exactly what the dashboard shows for the current controls.")
    files = (
        ("Scenario config (JSON)", json.dumps(cfg.to_dict(), indent=2), f"{stem}_config.json", "application/json"),
        ("Summary metrics (CSV)", comparison.summary_frame().to_csv(index=False), f"{stem}_summary.csv", "text/csv"),
        ("Per-microgrid results (CSV)", comparison.player_frame().to_csv(index=False), f"{stem}_players.csv",
         "text/csv"),
        ("Trades, all mechanisms (CSV)", comparison.trades_frame().to_csv(index=False), f"{stem}_trades.csv",
         "text/csv"),
        ("Settlements, all mechanisms (CSV)", comparison.settlement_frame().to_csv(index=False),
         f"{stem}_settlement.csv", "text/csv"),
        ("Solver diagnostics (CSV)", comparison.diagnostics_frame().to_csv(index=False), f"{stem}_diagnostics.csv",
         "text/csv"),
        ("Solver diagnostics (JSON)",
         json.dumps(diagnostics_records(r for _, rs in comparison.results for r in rs), indent=2),
         f"{stem}_diagnostics.json", "application/json"),
    )
    for i, (label, data, name, mime) in enumerate(files):
        st.download_button(label, data=data, file_name=name, mime=mime, key=f"download_{i}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="MultiGrid", layout="wide")
    pal = PALETTE[theme_name()]
    try:
        profiles, cfg_dict = sidebar_inputs()
        cfg_json = json.dumps(cfg_dict, sort_keys=True)
        comparison, notes = run_all(profiles, cfg_json)
    except (DataValidationError, EnergyBalanceError, SettlementError, SolverError) as exc:
        # Never show partial or unvalidated numbers: explain the failure and stop.
        st.error(f"The simulation could not produce a validated result, so no figures are shown. {exc}")
        st.stop()
    cfg = comparison.config
    colors = {mg: pal["series"][i] for i, mg in enumerate(sorted(cfg.microgrid_ids))}

    st.title("MultiGrid: P2P renewable energy trading among microgrids")
    st.caption(f"microgrid package v{__version__} · scenario {cfg.scenario_id}")
    st.warning(
        "Simulation only. Generation and demand profiles are synthetic and tariffs are illustrative assumptions, "
        "not real utility tariffs or real-time data. This is a research demonstration, not a trading or "
        "grid-control system."
    )
    for note in notes:
        st.warning(note)

    tabs = st.tabs(["Overview", "Profiles", "Simulate", "Transactions", "Game analysis", "Comparison", "Network",
                    "Downloads"])
    with tabs[0]:
        page_overview(comparison, profiles)
    with tabs[1]:
        page_profiles(profiles, cfg, pal, colors)
    with tabs[2]:
        page_simulate(comparison)
    with tabs[3]:
        page_transactions(comparison, pal)
    with tabs[4]:
        page_game(comparison, profiles, cfg_json, pal, colors)
    with tabs[5]:
        page_comparison(comparison, pal)
    with tabs[6]:
        page_network(comparison, pal, colors)
    with tabs[7]:
        page_downloads(comparison)


main()
