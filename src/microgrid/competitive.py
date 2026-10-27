"""Competitive market: a finite simultaneous seller price game with exhaustive pure-Nash verification.

The game (blueprint section 8)
    Players
        Every seller of the interval (S_i > tol). Buyers are not strategic.

    Strategies
        One posted gross price p_i from a finite grid P_i: multiples of
        ``price_step_inr`` with p_i - f_s >= a (individually rational for the
        seller) and p_i + f_b < b - tol (acceptable to a buyer). The lowest
        grid price is the seller's reservation price, where selling pays the
        same as exporting.

    Buyer rule (deterministic, non-strategic followers)
        1. Each offer's delivered price is p_i + f_b.
        2. Each buyer sorts acceptable offers by (delivered price, seller ID).
        3. Buyers are processed in ascending ID. Each takes as much as possible
           from its first offer, then the next, never exceeding the remaining
           seller surplus or its own deficit.
        4. An offer is acceptable only if the delivered price is < b - tol.
        5. Unserved deficit is imported from the grid; unsold surplus is exported.

    Payoff
        U_i(p) = sum_j (p_i - a - f_s) * q_ij(p), seller i's gain over
        exporting the same energy (not its sales revenue).

    Equilibrium
        p is a pure Nash equilibrium iff max_i regret_i(p) <= tol, where
            regret_i(p) = max over p_i' in P_i of U_i(p_i', p_-i) - U_i(p).
        The full payoff table is built (each profile simulated exactly
        once), and every unilateral deviation of every seller in every
        profile is evaluated. Nothing is sampled or approximated.

Reporting
    * Pure NE found. All equilibria are counted. The one with the lowest sum
      of prices (then the lexicographically smallest price vector) is
      selected and settled through the shared ``settlement.settle_market``.
    * No pure NE. Status ``NO_PURE_NE``. No P2P trade is settled for that
      interval, because the game makes no prediction there. The
      minimum-maximum-regret profile is reported in the diagnostics only,
      labelled NOT NASH.
    * More sellers than ``max_competitive_sellers``. Exact enumeration is
      not attempted. Status ``ENUMERATION_CAP_EXCEEDED`` is set, a
      ``CombinatorialWarning`` is emitted, and nothing is claimed or settled.

Scope notes
    This is a simultaneous-move game among sellers with a fixed buyer rule,
    not a bi-level Stackelberg equilibrium. The ID-based tie rule is a
    modelling choice that can change which profiles are equilibria (and who
    earns what at them), so results are not claimed to be independent of it.

Interaction
    Consumes ``HourState`` objects from ``energy`` and tariffs/config from
    ``models``. The selected equilibrium's trades go through the shared
    ``settlement.settle_market``, exactly like the other mechanisms.
"""

from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .data import validate_profiles
from .energy import compute_hour_states
from .models import (
    CombinatorialWarning,
    HourState,
    MarketDiagnostics,
    MarketResult,
    ScenarioConfig,
    TariffSchedule,
    Trade,
)
from .settlement import settle_market

MECHANISM = "competitive"
STATUS_SUCCESS = "success"
STATUS_NO_TRADE = "no_trade"
STATUS_NO_PURE_NE = "NO_PURE_NE"
STATUS_CAP_EXCEEDED = "ENUMERATION_CAP_EXCEEDED"

GAME_SOLVED = "solved"
GAME_NONE = "no_game"
GAME_CAP_EXCEEDED = "cap_exceeded"

_PRICE_DECIMALS = 10


# ---------------------------------------------------------------------------
# Strategy sets and the buyer rule
# ---------------------------------------------------------------------------


def price_grid(
    seller_export_price: float,
    buyer_retail_price: float,
    step: float,
    seller_fee: float = 0.0,
    buyer_fee: float = 0.0,
    tol: float = 1e-8,
) -> tuple[float, ...]:
    """Individually rational posted prices on a fixed tick.

    Returns every multiple of ``step`` with p - f_s >= a and p + f_b < b - tol.
    The default (a = 3, b = 8, no fees, step 0.5) is (3.0, 3.5, ..., 7.5).
    """
    if not (math.isfinite(step) and step > 0):
        raise ValueError(f"price step must be finite and > 0, got {step}")
    low = seller_export_price + seller_fee
    high = buyer_retail_price - buyer_fee
    prices = []
    k = math.ceil(low / step - 1e-9)
    while True:
        price = round(k * step, _PRICE_DECIMALS)
        if price >= high - tol:
            break
        if price >= low - tol:
            prices.append(price)
        k += 1
    return tuple(prices)


def _filter_grid(prices: Sequence[float], low: float, high: float, tol: float) -> tuple[float, ...]:
    """Keep only individually rational prices from a user-supplied grid (sorted, unique)."""
    cleaned = sorted({round(float(p), _PRICE_DECIMALS) for p in prices if math.isfinite(float(p))})
    return tuple(p for p in cleaned if low - tol <= p < high - tol)


def allocate_buyers(
    prices: Mapping[str, float],
    supply: Mapping[str, float],
    demand: Mapping[str, float],
    retail_price: float,
    buyer_fee: float = 0.0,
    tol: float = 1e-8,
) -> tuple[tuple[str, str, float], ...]:
    """Deterministic buyer response to posted seller prices.

    Returns (seller, buyer, kWh) in the order the trades were made. Buyers go
    in ascending ID order. Each takes acceptable offers (delivered price
    < retail - tol), cheapest first, ties broken by seller ID, and never
    exceeds the remaining seller surplus or its own deficit.
    """
    remaining = {seller: float(supply[seller]) for seller in prices}
    trades = []
    for buyer in sorted(demand):
        need = float(demand[buyer])
        offers = sorted(
            (prices[seller] + buyer_fee, seller)
            for seller in prices
            if prices[seller] + buyer_fee < retail_price - tol
        )
        for _, seller in offers:
            if need <= tol:
                break
            quantity = min(remaining[seller], need)
            if quantity <= tol:
                continue
            trades.append((seller, buyer, quantity))
            remaining[seller] -= quantity
            need -= quantity
    return tuple(trades)


def _selection_key(profile: tuple[float, ...]) -> tuple[float, tuple[float, ...]]:
    """Equilibrium selection: lowest sum of posted prices, then lexicographic price vector."""
    return (round(math.fsum(profile), 9), profile)


def _format_profile(sellers: Sequence[str], profile: Sequence[float]) -> str:
    return ", ".join(f"{s}={p:.2f}" for s, p in zip(sellers, profile))


# ---------------------------------------------------------------------------
# Game analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PriceGameAnalysis:
    """The complete payoff table and equilibrium analysis of one interval's seller price game.

    ``utilities`` has shape (|P_1|, ..., |P_m|, m), where entry [k_1..k_m, i]
    is seller i's payoff when each seller j posts its k_j-th price.
    ``max_regret`` has shape (|P_1|, ..., |P_m|). Both arrays are read-only.
    """

    day: int
    hour: int
    sellers: tuple[str, ...]
    buyers: tuple[str, ...]
    price_grids: tuple[tuple[float, ...], ...]
    status: str
    tol: float = 1e-8
    message: str = ""
    equilibria: tuple[tuple[float, ...], ...] = ()
    utilities: np.ndarray | None = field(default=None, compare=False, repr=False)
    max_regret: np.ndarray | None = field(default=None, compare=False, repr=False)

    @property
    def examined_profiles(self) -> int:
        """Number of joint price profiles whose payoffs and deviations were all evaluated."""
        return 0 if self.max_regret is None else int(self.max_regret.size)

    @property
    def pure_ne_found(self) -> bool | None:
        """True or False after exhaustive enumeration; None if no game was analysed."""
        return None if self.status != GAME_SOLVED else bool(self.equilibria)

    def _index(self, profile: Sequence[float]) -> tuple[int, ...]:
        if self.status != GAME_SOLVED:
            raise ValueError(f"no payoff table: game status is {self.status!r}")
        if len(profile) != len(self.sellers):
            raise ValueError(f"profile needs one price per seller {self.sellers}, got {tuple(profile)}")
        index = []
        for seller, grid, price in zip(self.sellers, self.price_grids, profile):
            matches = [k for k, p in enumerate(grid) if abs(p - float(price)) <= 1e-9]
            if not matches:
                raise ValueError(f"price {price} is not in {seller}'s strategy set {grid}")
            index.append(matches[0])
        return tuple(index)

    def payoffs(self, profile: Sequence[float]) -> dict[str, float]:
        """Each seller's payoff U_i at a joint price profile."""
        row = self.utilities[self._index(profile)]
        return {seller: float(row[i]) for i, seller in enumerate(self.sellers)}

    def maximum_unilateral_regret(self, profile: Sequence[float]) -> float:
        """max_i [best payoff from a unilateral deviation - payoff at the profile]."""
        return float(self.max_regret[self._index(profile)])

    @property
    def selected_equilibrium(self) -> tuple[float, ...] | None:
        return min(self.equilibria, key=_selection_key) if self.equilibria else None

    @property
    def min_max_regret_profile(self) -> tuple[float, ...] | None:
        """The profile with the smallest maximum regret: a diagnostic only, NOT NASH when regret > tol."""
        if self.max_regret is None:
            return None
        floor = float(self.max_regret.min())
        candidates = [
            tuple(self.price_grids[i][k] for i, k in enumerate(idx))
            for idx in np.argwhere(self.max_regret <= floor + self.tol)
        ]
        return min(candidates, key=_selection_key)


def analyze_price_game(
    state: HourState,
    cfg: ScenarioConfig,
    tariffs: TariffSchedule | None = None,
    price_grid_override: Sequence[float] | None = None,
) -> PriceGameAnalysis:
    """Build the full payoff table of the interval's seller price game and find every pure NE.

    ``price_grid_override`` replaces the tick grid (it is still filtered to
    individually rational prices). It is used for unit tests such as the
    blueprint's one-seller grid {3, 4, 5, 6, 7, 7.5}.
    """
    schedule = tariffs if tariffs is not None else TariffSchedule.from_config(cfg)
    retail = schedule.retail_at(state.hour)
    export = schedule.export_at(state.hour)
    f_s, f_b, tol = cfg.seller_fee_inr_per_kwh, cfg.buyer_fee_inr_per_kwh, cfg.tol
    sellers, buyers = state.sellers, state.buyers
    common = dict(day=state.day, hour=state.hour, sellers=sellers, buyers=buyers, tol=tol)

    if not sellers or not buyers:
        return PriceGameAnalysis(
            price_grids=(), status=GAME_NONE, message="no game: the interval has no sellers or no buyers", **common
        )
    if price_grid_override is None:
        grid = price_grid(export, retail, cfg.price_step_inr, f_s, f_b, tol)
    else:
        grid = _filter_grid(price_grid_override, export + f_s, retail - f_b, tol)
    if not grid:
        return PriceGameAnalysis(
            price_grids=(),
            status=GAME_NONE,
            message=(
                f"no game: no individually rational price (seller needs p >= a + f_s = {export + f_s:.4f}, "
                f"buyer needs p < b - f_b = {retail - f_b:.4f} INR/kWh)"
            ),
            **common,
        )
    grids = tuple(grid for _ in sellers)
    profile_count = math.prod(len(g) for g in grids)
    if len(sellers) > cfg.max_competitive_sellers:
        return PriceGameAnalysis(
            price_grids=grids,
            status=GAME_CAP_EXCEEDED,
            message=(
                f"{len(sellers)} sellers exceed max_competitive_sellers={cfg.max_competitive_sellers} "
                f"({profile_count} joint profiles); exact enumeration not attempted and no equilibrium is claimed"
            ),
            **common,
        )

    supply = {s: state.player(s).surplus_kwh for s in sellers}
    demand = {b: state.player(b).deficit_kwh for b in buyers}
    shape = tuple(len(g) for g in grids)
    utilities = np.zeros(shape + (len(sellers),))
    for index in itertools.product(*(range(n) for n in shape)):
        prices = {s: grids[i][index[i]] for i, s in enumerate(sellers)}
        sold = dict.fromkeys(sellers, 0.0)
        for seller, _, quantity in allocate_buyers(prices, supply, demand, retail, f_b, tol):
            sold[seller] += quantity
        utilities[index] = [(prices[s] - export - f_s) * sold[s] for s in sellers]

    max_regret = np.zeros(shape)
    for i in range(len(sellers)):
        own = utilities[..., i]
        best_deviation = own.max(axis=i, keepdims=True)  # best response to the others' prices
        max_regret = np.maximum(max_regret, best_deviation - own)
    equilibria = tuple(
        tuple(grids[i][k] for i, k in enumerate(idx)) for idx in np.argwhere(max_regret <= tol)
    )
    utilities.setflags(write=False)
    max_regret.setflags(write=False)
    return PriceGameAnalysis(
        price_grids=grids,
        status=GAME_SOLVED,
        message=f"exhaustive enumeration of {profile_count} profiles; {len(equilibria)} pure NE",
        equilibria=equilibria,
        utilities=utilities,
        max_regret=max_regret,
        **common,
    )


# ---------------------------------------------------------------------------
# Mechanism entry points
# ---------------------------------------------------------------------------


def solve_competitive(
    state: HourState,
    cfg: ScenarioConfig,
    tariffs: TariffSchedule | None = None,
    price_grid_override: Sequence[float] | None = None,
) -> MarketResult:
    """Competitive market for one interval: analyse the price game and settle a verified pure NE.

    Only a verified pure Nash equilibrium is ever settled. If none exists,
    or exact enumeration is not possible, the interval is settled without
    P2P trades and the diagnostics say why.
    """
    schedule = tariffs if tariffs is not None else TariffSchedule.from_config(cfg)
    analysis = analyze_price_game(state, cfg, schedule, price_grid_override)

    def settle(trades, diagnostics):
        return settle_market(state, trades, cfg, mechanism=MECHANISM, tariffs=schedule, diagnostics=diagnostics)

    if analysis.status == GAME_NONE:
        return settle((), MarketDiagnostics(status=STATUS_NO_TRADE, examined_profiles=0, message=analysis.message))

    if analysis.status == GAME_CAP_EXCEEDED:
        warnings.warn(f"day {state.day} hour {state.hour}: {analysis.message}", CombinatorialWarning, stacklevel=2)
        return settle(
            (), MarketDiagnostics(status=STATUS_CAP_EXCEEDED, examined_profiles=0, message=analysis.message)
        )

    if not analysis.equilibria:
        diagnostic = analysis.min_max_regret_profile
        regret = analysis.maximum_unilateral_regret(diagnostic)
        return settle(
            (),
            MarketDiagnostics(
                status=STATUS_NO_PURE_NE,
                pure_ne_found=False,
                equilibrium_count=0,
                examined_profiles=analysis.examined_profiles,
                max_regret_inr=regret,
                not_nash_profile=tuple(zip(analysis.sellers, diagnostic)),
                message=(
                    f"NO PURE EQUILIBRIUM FOUND after checking every unilateral deviation in "
                    f"{analysis.examined_profiles} profiles; no P2P trade is settled. Diagnostic "
                    f"minimum-maximum-regret profile ({_format_profile(analysis.sellers, diagnostic)}) "
                    f"is NOT NASH: max regret INR {regret:.4f}"
                ),
            ),
        )

    selected = analysis.selected_equilibrium
    prices = dict(zip(analysis.sellers, selected))
    supply = {s: state.player(s).surplus_kwh for s in analysis.sellers}
    demand = {b: state.player(b).deficit_kwh for b in analysis.buyers}
    trades = tuple(
        Trade(
            day=state.day,
            hour=state.hour,
            seller_id=seller,
            buyer_id=buyer,
            quantity_kwh=quantity,
            price_inr_per_kwh=prices[seller],
            seller_fee_inr_per_kwh=cfg.seller_fee_inr_per_kwh,
            buyer_fee_inr_per_kwh=cfg.buyer_fee_inr_per_kwh,
        )
        for seller, buyer, quantity in allocate_buyers(
            prices, supply, demand, schedule.retail_at(state.hour), cfg.buyer_fee_inr_per_kwh, cfg.tol
        )
    )
    count = len(analysis.equilibria)
    return settle(
        trades,
        MarketDiagnostics(
            status=STATUS_SUCCESS if trades else STATUS_NO_TRADE,
            pure_ne_found=True,
            equilibrium_count=count,
            examined_profiles=analysis.examined_profiles,
            max_regret_inr=analysis.maximum_unilateral_regret(selected),
            selected_profile=tuple(zip(analysis.sellers, selected)),
            message=(
                f"verified pure NE ({_format_profile(analysis.sellers, selected)}); {count} pure NE among "
                f"{analysis.examined_profiles} profiles; selection: lowest price sum, then lexicographic"
            ),
        ),
    )


def run_competitive(
    profiles: pd.DataFrame,
    cfg: ScenarioConfig,
    tariffs: TariffSchedule | None = None,
) -> list[MarketResult]:
    """Validate the profiles, then run the competitive market for every interval in time order.

    The input frame is not modified.
    """
    validated = validate_profiles(profiles, cfg)
    schedule = tariffs if tariffs is not None else TariffSchedule.from_config(cfg)
    return [solve_competitive(state, cfg, schedule) for state in compute_hour_states(validated, cfg.tol)]
