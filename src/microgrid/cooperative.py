"""Cooperative market: welfare-maximising LP allocation plus bilateral weighted Nash bargaining.

Stage A: allocation (blueprint section 7)
    For every seller i (S_i > tol) and buyer j (R_j > tol), the unit trade
    surplus is

        g_ij = b_j - a_i - f_s - f_b

    Only edges with g_ij > tol are eligible. The social LP is

        maximise   sum g_ij * q_ij
        subject to sum_j q_ij <= S_i,   sum_i q_ij <= R_j,   q_ij >= 0

    It is solved with ``scipy.optimize.linprog(method="highs")``. Every solver
    call is status-checked, and a failure raises ``SolverError``; no result
    is ever reported for a failed solve.

Tie rule (deterministic, documented)
    The LP usually has many optimal allocations. In the MVP every edge has
    the same g, because tariffs and fees are uniform within an hour. Among
    the optimal allocations we report the lexicographic maximum of q in
    (seller_id, buyer_id) order.

    The set of optimal allocations (the optimal face) is identified exactly
    from the primary LP's dual solution, by complementary slackness: a
    capacity row with a nonzero dual is at capacity in every optimum, and an
    edge with a positive reduced cost is zero in every optimum. One extra LP
    per edge then maximises that edge's quantity over the optimal face, with
    earlier edges fixed. No welfare slack is used, so the tie-break cannot
    trade welfare for priority. The realised welfare is re-checked against
    the optimum.

    When g is uniform on a complete seller-buyer graph, this allocation
    equals the transparent sorted greedy rule (seller ID, then buyer ID).
    The greedy rule is computed independently, and any disagreement raises
    ``SolverError``.

Stage B: price by bilateral weighted Nash bargaining
    Each allocated quantity q_ij is fixed while its price is bargained. The
    disagreement point is trading that same quantity with the utility instead:

        r_s = a_i + f_s   (seller reservation price)
        r_b = b_j - f_b   (buyer's maximum acceptable gross price)
        dU_s = q (p - r_s),   dU_b = q (r_b - p)
        p* = argmax dU_s^alpha * dU_b^(1 - alpha) = (1 - alpha) * r_s + alpha * r_b

    alpha = ``bargaining_seller_weight`` in (0, 1). The seller receives the
    share alpha of each trade's surplus, so alpha = 0.5 splits it equally.

Scope note
    This is *bilateral* bargaining on a centrally allocated, fixed quantity.
    It is not multilateral Nash bargaining. The LP allocation is an
    optimisation benchmark, not a Nash equilibrium.

Interaction
    Consumes ``HourState`` objects from ``energy`` and tariffs/config from
    ``models``. Produces ``Trade`` objects and hands them to the shared
    ``settlement.settle_market``, which derives and validates every flow and
    payment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .data import validate_profiles
from .energy import compute_hour_states
from .models import (
    HourState,
    MarketDiagnostics,
    MarketResult,
    ScenarioConfig,
    SettlementError,
    SolverError,
    TariffSchedule,
    Trade,
)
from .settlement import settle_market

MECHANISM = "cooperative"
LP_METHOD = "highs"
#: Relative tolerance for accepting LP output (HiGHS's default primal feasibility tolerance).
LP_RELATIVE_TOL = 1e-7
#: Dual values above this (relative to the largest g) mark a binding row or a zero-forced edge.
DUAL_TOL = 1e-9
#: LP quantities are rounded to 1e-9 kWh to strip solver float noise (14.999999999999996 -> 15.0).
#: Any capacity excess this creates is trimmed by the repair step.
ALLOCATION_DECIMALS = 9


# ---------------------------------------------------------------------------
# Bilateral economics
# ---------------------------------------------------------------------------


def unit_trade_surplus(
    buyer_retail_price: float,
    seller_export_price: float,
    seller_fee: float = 0.0,
    buyer_fee: float = 0.0,
) -> float:
    """g = b_j - a_i - f_s - f_b: the community gain per kWh moved from seller i to buyer j."""
    return buyer_retail_price - seller_export_price - seller_fee - buyer_fee


def is_beneficial_trade(
    buyer_retail_price: float,
    seller_export_price: float,
    seller_fee: float = 0.0,
    buyer_fee: float = 0.0,
    tol: float = 1e-8,
) -> bool:
    """True only if the trade has strictly positive surplus (g > tol)."""
    return unit_trade_surplus(buyer_retail_price, seller_export_price, seller_fee, buyer_fee) > tol


def bargaining_price(
    seller_export_price: float,
    buyer_retail_price: float,
    seller_fee: float = 0.0,
    buyer_fee: float = 0.0,
    alpha: float = 0.5,
    tol: float = 1e-8,
) -> float:
    """Gross price p* = (1 - alpha) * (a + f_s) + alpha * (b - f_b) maximising the weighted Nash product.

    Raises ``ValueError`` if alpha is not strictly between 0 and 1, or if
    there is no positive bargaining surplus (r_b - r_s <= tol). Such a trade
    is rejected rather than given an arbitrary price.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"bargaining weight alpha must be strictly between 0 and 1, got {alpha}")
    seller_reservation = seller_export_price + seller_fee
    buyer_reservation = buyer_retail_price - buyer_fee
    if buyer_reservation - seller_reservation <= tol:
        raise ValueError(
            f"no positive bargaining surplus: r_b - r_s = {buyer_reservation - seller_reservation} "
            f"<= tol {tol}; the trade must be rejected"
        )
    return (1.0 - alpha) * seller_reservation + alpha * buyer_reservation


def nash_product(
    price: float,
    quantity_kwh: float,
    seller_export_price: float,
    buyer_retail_price: float,
    seller_fee: float = 0.0,
    buyer_fee: float = 0.0,
    alpha: float = 0.5,
) -> float:
    """Weighted Nash product dU_s^alpha * dU_b^(1-alpha). Returns 0 when either side would not gain."""
    gain_seller = quantity_kwh * (price - seller_export_price - seller_fee)
    gain_buyer = quantity_kwh * (buyer_retail_price - buyer_fee - price)
    if gain_seller <= 0.0 or gain_buyer <= 0.0:
        return 0.0
    return gain_seller**alpha * gain_buyer ** (1.0 - alpha)


# ---------------------------------------------------------------------------
# Stage A: allocation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Edge:
    """An eligible seller-buyer pair and its unit trade surplus g (INR/kWh)."""

    seller_id: str
    buyer_id: str
    unit_surplus_inr_per_kwh: float


@dataclass(frozen=True)
class Allocation:
    """Welfare-maximising quantities, one per edge, with edges sorted by (seller, buyer)."""

    edges: tuple[Edge, ...]
    quantities_kwh: tuple[float, ...]
    optimal_welfare_inr: float
    lp_solves: int
    cross_check: str

    def quantity(self, seller_id: str, buyer_id: str) -> float:
        for edge, q in zip(self.edges, self.quantities_kwh):
            if (edge.seller_id, edge.buyer_id) == (seller_id, buyer_id):
                return q
        return 0.0

    @property
    def total_kwh(self) -> float:
        return math.fsum(self.quantities_kwh)

    @property
    def welfare_inr(self) -> float:
        """Realised welfare sum(g * q) of the reported allocation."""
        return math.fsum(e.unit_surplus_inr_per_kwh * q for e, q in zip(self.edges, self.quantities_kwh))


def eligible_edges(
    state: HourState, cfg: ScenarioConfig, tariffs: TariffSchedule | None = None
) -> tuple[Edge, ...]:
    """All seller-buyer pairs of the interval with unit surplus g > tol, sorted by (seller, buyer)."""
    schedule = tariffs if tariffs is not None else TariffSchedule.from_config(cfg)
    retail = schedule.retail_at(state.hour)
    export = schedule.export_at(state.hour)
    edges = []
    for seller in state.sellers:
        for buyer in state.buyers:
            g = unit_trade_surplus(retail, export, cfg.seller_fee_inr_per_kwh, cfg.buyer_fee_inr_per_kwh)
            if g > cfg.tol:
                edges.append(Edge(seller, buyer, g))
    return tuple(edges)


def greedy_allocation(
    supply: Mapping[str, float],
    demand: Mapping[str, float],
    edges: Sequence[Edge],
    tol: float = 1e-8,
) -> tuple[float, ...]:
    """Sorted greedy rule: for each seller by ID, fill buyers by ID with min(remaining supply, remaining demand).

    Returns quantities aligned with ``edges`` sorted by (seller, buyer). It
    is used as an independent cross-check of the LP.
    """
    remaining_supply = {k: float(v) for k, v in supply.items()}
    remaining_demand = {k: float(v) for k, v in demand.items()}
    quantities = []
    for edge in sorted(edges, key=lambda e: (e.seller_id, e.buyer_id)):
        q = min(remaining_supply[edge.seller_id], remaining_demand[edge.buyer_id])
        q = q if q > tol else 0.0
        remaining_supply[edge.seller_id] -= q
        remaining_demand[edge.buyer_id] -= q
        quantities.append(q)
    return tuple(quantities)


def _solve_lp(
    c: np.ndarray,
    a_ub: np.ndarray,
    b_ub: np.ndarray,
    bounds: list,
    stage: str,
    a_eq: np.ndarray | None = None,
    b_eq: np.ndarray | None = None,
):
    def _rows(matrix, rhs):
        return (None, None) if matrix is None or matrix.shape[0] == 0 else (matrix, rhs)

    a_ub, b_ub = _rows(a_ub, b_ub)
    a_eq, b_eq = _rows(a_eq, b_eq)
    result = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method=LP_METHOD)
    if result.status != 0 or not result.success or result.x is None:
        raise SolverError(
            f"HiGHS LP failed during {stage}: status {result.status} ({result.message}); no allocation is reported"
        )
    return result


def _repair_capacities(
    q: np.ndarray, a_ub: np.ndarray, capacity: np.ndarray, row_names: list[str], limit: float
) -> np.ndarray:
    """Remove float-level capacity excess (<= limit) by trimming that row's last edges.

    Trimming a quantity can never violate another <= constraint. Any excess
    above ``limit`` is a solver failure.
    """
    q = q.copy()
    for row, name in enumerate(row_names):
        excess = float(a_ub[row] @ q - capacity[row])
        if excess <= 0.0:
            continue
        if excess > limit:
            raise SolverError(f"LP allocation exceeds {name} by {excess} kWh (limit {limit})")
        for k in reversed(np.flatnonzero(a_ub[row])):
            cut = min(q[k], excess)
            q[k] -= cut
            excess -= cut
            if excess <= 0.0:
                break
    return q


def allocate_max_welfare(
    supply: Mapping[str, float],
    demand: Mapping[str, float],
    edges: Sequence[Edge],
    tol: float = 1e-8,
) -> Allocation:
    """Solve the social LP, apply the lexicographic tie rule and cross-check against greedy."""
    ordered = tuple(sorted(edges, key=lambda e: (e.seller_id, e.buyer_id)))
    pairs = [(e.seller_id, e.buyer_id) for e in ordered]
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"duplicate edges: {pairs}")
    for edge in ordered:
        if edge.seller_id not in supply or edge.buyer_id not in demand:
            raise ValueError(f"edge {edge.seller_id}->{edge.buyer_id} has no matching supply/demand entry")
        if not (math.isfinite(edge.unit_surplus_inr_per_kwh) and edge.unit_surplus_inr_per_kwh > tol):
            raise ValueError(
                f"edge {edge.seller_id}->{edge.buyer_id} must have unit surplus > tol, "
                f"got {edge.unit_surplus_inr_per_kwh}"
            )
    for label, capacities in (("supply", supply), ("demand", demand)):
        for mg, value in capacities.items():
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"{label} of {mg} must be finite and >= 0, got {value}")
    if not ordered:
        return Allocation((), (), 0.0, 0, "no eligible edges")

    sellers = sorted({e.seller_id for e in ordered})
    buyers = sorted({e.buyer_id for e in ordered})
    n = len(ordered)
    a_ub = np.zeros((len(sellers) + len(buyers), n))
    for k, edge in enumerate(ordered):
        a_ub[sellers.index(edge.seller_id), k] = 1.0
        a_ub[len(sellers) + buyers.index(edge.buyer_id), k] = 1.0
    capacity = np.array([float(supply[s]) for s in sellers] + [float(demand[b]) for b in buyers])
    row_names = [f"seller {s} surplus" for s in sellers] + [f"buyer {b} deficit" for b in buyers]
    g = np.array([e.unit_surplus_inr_per_kwh for e in ordered])

    primary = _solve_lp(-g, a_ub, capacity, [(0.0, None)] * n, "welfare maximisation")
    optimum = float(g @ primary.x)

    # Optimal face from complementary slackness. linprog minimises -g.q, so a
    # binding capacity row has a negative marginal, and an edge that is zero
    # in every optimum has a positive lower-bound marginal (reduced cost).
    row_duals = getattr(getattr(primary, "ineqlin", None), "marginals", None)
    edge_duals = getattr(getattr(primary, "lower", None), "marginals", None)
    if row_duals is None or edge_duals is None:
        raise SolverError("HiGHS returned no dual values; the optimal face cannot be identified")
    dual_tol = DUAL_TOL * max(1.0, float(g.max()))
    binding = np.asarray(row_duals, dtype=float) < -dual_tol
    forced_zero = np.asarray(edge_duals, dtype=float) > dual_tol

    q = np.zeros(n)
    for k, edge in enumerate(ordered):
        bounds = [(q[j], q[j]) for j in range(k)] + [
            (0.0, 0.0) if forced_zero[j] else (0.0, None) for j in range(k, n)
        ]
        objective = np.zeros(n)
        objective[k] = -1.0
        step = _solve_lp(
            objective, a_ub[~binding], capacity[~binding], bounds,
            f"tie-break step {k + 1}/{n} (edge {edge.seller_id}->{edge.buyer_id})",
            a_eq=a_ub[binding], b_eq=capacity[binding],
        )
        q[k] = max(float(step.x[k]), 0.0)

    scale = max(1.0, float(capacity.max()))
    q = np.round(q, ALLOCATION_DECIMALS)
    q = _repair_capacities(q, a_ub, capacity, row_names, LP_RELATIVE_TOL * scale)
    q[q <= tol] = 0.0
    realised = float(g @ q)
    welfare_tol = LP_RELATIVE_TOL * max(1.0, abs(optimum))
    if realised < optimum - welfare_tol:
        raise SolverError(f"tie-break allocation welfare {realised} is below the LP optimum {optimum}")

    greedy = np.array(greedy_allocation(supply, demand, ordered, tol))
    if float(g.max() - g.min()) <= tol:
        if not np.allclose(q, greedy, rtol=0.0, atol=LP_RELATIVE_TOL * scale):
            raise SolverError(
                f"LP tie-break allocation {q.tolist()} differs from the sorted greedy allocation "
                f"{greedy.tolist()} although every edge has the same unit surplus"
            )
        cross_check = "LP tie-break allocation equals the sorted greedy allocation"
    else:
        if float(g @ greedy) > optimum + welfare_tol:
            raise SolverError("greedy allocation exceeds the LP optimum; the LP result is not optimal")
        cross_check = "greedy welfare does not exceed the LP optimum"

    return Allocation(ordered, tuple(float(x) for x in q), optimum, n + 1, cross_check)


# ---------------------------------------------------------------------------
# Mechanism entry points
# ---------------------------------------------------------------------------


def solve_cooperative(
    state: HourState,
    cfg: ScenarioConfig,
    tariffs: TariffSchedule | None = None,
) -> MarketResult:
    """Cooperative market for one interval: LP allocation, Nash-bargained prices, shared settlement.

    With no eligible edge (no sellers, no buyers, or b - a - f_s - f_b <= tol)
    it returns an empty trade log with a baseline-equivalent settlement and
    diagnostics status "no_trade".
    """
    schedule = tariffs if tariffs is not None else TariffSchedule.from_config(cfg)
    retail = schedule.retail_at(state.hour)
    export = schedule.export_at(state.hour)
    f_s = cfg.seller_fee_inr_per_kwh
    f_b = cfg.buyer_fee_inr_per_kwh
    alpha = cfg.bargaining_seller_weight

    edges = eligible_edges(state, cfg, schedule)
    if not edges:
        if not state.sellers or not state.buyers:
            reason = "no P2P trade: the interval has no sellers or no buyers"
        else:
            g = unit_trade_surplus(retail, export, f_s, f_b)
            reason = f"no P2P trade: unit surplus b - a - f_s - f_b = {g:.4f} INR/kWh is not positive"
        return settle_market(
            state, (), cfg, mechanism=MECHANISM, tariffs=schedule,
            diagnostics=MarketDiagnostics(status="no_trade", optimal_welfare_inr=0.0, message=reason),
        )

    supply = {mg: state.player(mg).surplus_kwh for mg in state.sellers}
    demand = {mg: state.player(mg).deficit_kwh for mg in state.buyers}
    allocation = allocate_max_welfare(supply, demand, edges, cfg.tol)
    trades = tuple(
        Trade(
            day=state.day,
            hour=state.hour,
            seller_id=edge.seller_id,
            buyer_id=edge.buyer_id,
            quantity_kwh=q,
            price_inr_per_kwh=bargaining_price(export, retail, f_s, f_b, alpha, cfg.tol),
            seller_fee_inr_per_kwh=f_s,
            buyer_fee_inr_per_kwh=f_b,
        )
        for edge, q in zip(allocation.edges, allocation.quantities_kwh)
        if q > cfg.tol
    )
    diagnostics = MarketDiagnostics(
        status="success" if trades else "no_trade",
        optimal_welfare_inr=allocation.optimal_welfare_inr,
        message=(
            f"HiGHS LP optimum INR {allocation.optimal_welfare_inr:.6f} ({allocation.lp_solves} LP solves); "
            f"tie rule: lexicographic (seller, buyer); cross-check: {allocation.cross_check}; "
            f"price: bilateral weighted Nash bargaining, alpha={alpha}"
        ),
    )
    result = settle_market(state, trades, cfg, mechanism=MECHANISM, tariffs=schedule, diagnostics=diagnostics)

    # Bargaining only redistributes surplus. If the settled community gain is
    # not the LP welfare, fees or payments were double-counted somewhere.
    if abs(result.total_gain_inr - allocation.welfare_inr) > cfg.tol * max(1.0, abs(allocation.welfare_inr)):
        raise SettlementError(
            f"settled community gain {result.total_gain_inr} INR differs from LP welfare "
            f"{allocation.welfare_inr} INR"
        )
    return result


def run_cooperative(
    profiles: pd.DataFrame,
    cfg: ScenarioConfig,
    tariffs: TariffSchedule | None = None,
) -> list[MarketResult]:
    """Validate the profiles, then run the cooperative market for every interval in time order.

    The input frame is not modified.
    """
    validated = validate_profiles(profiles, cfg)
    schedule = tariffs if tariffs is not None else TariffSchedule.from_config(cfg)
    return [solve_cooperative(state, cfg, schedule) for state in compute_hour_states(validated, cfg.tol)]
