"""Comparative metrics.

Implemented (Phase 2)
    ``jain_fairness``: the Jain index of non-negative gains, needed for the
    golden cooperative fixture (0.94736842).

Planned (Phase 4, blueprint section 9; NOT IMPLEMENTED)
    ``compute_metrics``: total P2P volume, net community cost, total and
    per-player gain, volume-weighted average price (N/A when volume = 0),
    efficiency relative to the LP optimum, cost reduction % (only when the
    baseline cost > tol), maximum seller regret (competitive mechanism
    only) and runtime. Every zero denominator is handled explicitly.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

from .models import MarketResult


def jain_fairness(gains: Iterable[float], tol: float = 1e-8) -> float | None:
    """Jain fairness index (sum x)^2 / (n * sum x^2) of per-player gains.

    It is defined for non-negative gains: 1.0 means equal gains, and 1/n
    means one player takes everything. Returns ``None`` (N/A) when the list
    is empty or every gain is <= tol, so there is never a division by zero.
    A gain below -tol raises ``ValueError``: a negative gain violates
    voluntary participation and must not be fed into this formula.
    """
    values = [float(g) for g in gains]
    if not values:
        return None
    if any(not math.isfinite(v) for v in values):
        raise ValueError(f"gains must be finite, got {values}")
    negative = [v for v in values if v < -tol]
    if negative:
        raise ValueError(
            f"Jain fairness is defined for non-negative gains; got {negative} "
            "(a negative gain violates voluntary participation)"
        )
    clipped = [max(v, 0.0) for v in values]
    if all(v <= tol for v in clipped):
        return None
    total = math.fsum(clipped)
    return total * total / (len(clipped) * math.fsum(v * v for v in clipped))


METRIC_COLUMNS = (
    "scenario_id",
    "mechanism",
    "intervals",
    "trading_intervals",
    "baseline_cost_inr",
    "net_community_cost_inr",
    "total_gain_inr",
    "cost_reduction_pct",
    "total_p2p_volume_kwh",
    "average_price_inr_per_kwh",
    "operator_fee_revenue_inr",
    "jain_fairness",
    "lp_optimal_gain_inr",
    "efficiency",
    "pure_ne_intervals",
    "no_pure_ne_intervals",
    "cap_exceeded_intervals",
    "max_regret_inr",
    "max_not_nash_regret_inr",
    "runtime_seconds",
)


def compute_metrics(
    results: Sequence[MarketResult],
    *,
    lp_optimal_gain_inr: float | None = None,
    runtime_seconds: float | None = None,
    tol: float = 1e-8,
) -> dict[str, Any]:
    """Summary metrics for one mechanism's results (blueprint section 9).

    Every zero denominator is explicit: ``None`` means N/A.
        average price   N/A when P2P volume <= tol
        efficiency      realised gain / LP-optimal gain; N/A when the optimum <= tol or is unknown
        cost reduction  only when the baseline cost > tol (otherwise read total_gain_inr)
        Jain fairness   N/A when every gain is zero
        NE counts and regrets: competitive mechanism only (N/A otherwise)
    For the cooperative mechanism the LP optimum is read from its own diagnostics if not given.
    """
    results = list(results)
    if not results:
        raise ValueError("compute_metrics needs at least one result")
    mechanisms = {r.mechanism for r in results}
    scenarios = {r.scenario_id for r in results}
    if len(mechanisms) != 1 or len(scenarios) != 1:
        raise ValueError(f"results must come from one mechanism and scenario, got {mechanisms} / {scenarios}")
    mechanism = mechanisms.pop()

    trades = [t for r in results for t in r.trades]
    volume = math.fsum(t.quantity_kwh for t in trades)
    baseline = math.fsum(r.total_baseline_cost_inr for r in results)
    cost = math.fsum(r.total_cost_inr for r in results)
    gain = math.fsum(r.total_gain_inr for r in results)
    per_player: dict[str, list[float]] = {}
    for r in results:
        for s in r.settlements:
            per_player.setdefault(s.microgrid_id, []).append(s.gain_inr)
    per_player_gain = {mg: math.fsum(v) for mg, v in sorted(per_player.items())}

    if lp_optimal_gain_inr is None and mechanism == "cooperative":
        lp_optimal_gain_inr = math.fsum(r.diagnostics.optimal_welfare_inr or 0.0 for r in results)
    efficiency = gain / lp_optimal_gain_inr if lp_optimal_gain_inr is not None and lp_optimal_gain_inr > tol else None

    competitive = mechanism == "competitive"
    ne_regrets = [r.diagnostics.max_regret_inr for r in results if r.diagnostics.pure_ne_found]
    no_ne_regrets = [r.diagnostics.max_regret_inr for r in results if r.diagnostics.status == "NO_PURE_NE"]
    return {
        "scenario_id": scenarios.pop(),
        "mechanism": mechanism,
        "intervals": len(results),
        "trading_intervals": sum(1 for r in results if r.trades),
        "baseline_cost_inr": baseline,
        "net_community_cost_inr": cost,
        "total_gain_inr": gain,
        "cost_reduction_pct": 100.0 * (baseline - cost) / baseline if baseline > tol else None,
        "total_p2p_volume_kwh": volume,
        "average_price_inr_per_kwh": math.fsum(t.payment_inr for t in trades) / volume if volume > tol else None,
        "operator_fee_revenue_inr": math.fsum(t.fee_revenue_inr for t in trades),
        "jain_fairness": jain_fairness(per_player_gain.values(), tol=tol),
        "lp_optimal_gain_inr": lp_optimal_gain_inr,
        "efficiency": efficiency,
        "pure_ne_intervals": len(ne_regrets) if competitive else None,
        "no_pure_ne_intervals": len(no_ne_regrets) if competitive else None,
        "cap_exceeded_intervals": (
            sum(1 for r in results if r.diagnostics.status == "ENUMERATION_CAP_EXCEEDED") if competitive else None
        ),
        "max_regret_inr": max(ne_regrets) if competitive and ne_regrets else None,
        "max_not_nash_regret_inr": max(no_ne_regrets) if competitive and no_ne_regrets else None,
        "runtime_seconds": runtime_seconds,
        "per_player_gain_inr": per_player_gain,
    }
