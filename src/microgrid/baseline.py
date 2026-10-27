"""Utility-only (no P2P) electricity market.

This is the reference outcome for every later mechanism: the disagreement
point for cooperative bargaining and the outside option for the competitive
game. Each microgrid imports its whole deficit from the utility at the
retail price b[t] and exports its whole surplus at the export price a[t]:

    B[i,t] = R[i,t]        X[i,t] = S[i,t]
    C_base[i,t] = b[t] * R[i,t] - a[t] * S[i,t]       (signed INR)

Interaction
    Takes ``HourState`` objects from ``energy`` and tariffs from
    ``models.TariffSchedule`` (built from ``ScenarioConfig`` by default).
    Settles through the shared ``settlement.settle_market`` with an empty
    trade log, so the baseline and the P2P mechanisms use identical
    accounting. Returns ``MarketResult`` objects, which ``data`` turns into
    CSV tables.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import pandas as pd

from .data import results_to_settlement_frame, validate_profiles
from .energy import compute_hour_states
from .models import HourState, MarketDiagnostics, MarketResult, ScenarioConfig, TariffSchedule
from .settlement import baseline_cost, settle_market

__all__ = [
    "MECHANISM",
    "baseline_cost",
    "community_baseline_cost",
    "run_baseline",
    "solve_baseline",
    "summarize_baseline",
]

MECHANISM = "baseline"


def solve_baseline(
    state: HourState,
    cfg: ScenarioConfig,
    tariffs: TariffSchedule | None = None,
) -> MarketResult:
    """Settle one interval with no P2P trading: ``settle_market(state, trades=())``.

    The shared settlement validates every player's energy balance and money
    before returning, so only validated settlements reach the reporting layer.
    """
    return settle_market(
        state,
        (),
        cfg,
        mechanism=MECHANISM,
        tariffs=tariffs,
        diagnostics=MarketDiagnostics(status="success", message="utility-only baseline: no P2P trades"),
    )


def run_baseline(
    profiles: pd.DataFrame,
    cfg: ScenarioConfig,
    tariffs: TariffSchedule | None = None,
) -> list[MarketResult]:
    """Validate the profiles, then run the baseline for every interval in time order.

    The input frame is not modified.
    """
    validated = validate_profiles(profiles, cfg)
    schedule = tariffs if tariffs is not None else TariffSchedule.from_config(cfg)
    return [solve_baseline(state, cfg, schedule) for state in compute_hour_states(validated, cfg.tol)]


def community_baseline_cost(results: Iterable[MarketResult]) -> float:
    """Sum of signed baseline costs over every player and interval (INR)."""
    return math.fsum(r.total_baseline_cost_inr for r in results)


def summarize_baseline(results: Sequence[MarketResult]) -> pd.DataFrame:
    """Per-microgrid totals over all intervals: grid import/export (kWh) and baseline cost (INR)."""
    columns = ["microgrid_id", "grid_import_kwh", "grid_export_kwh", "baseline_cost_inr"]
    frame = results_to_settlement_frame(results)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    return (
        frame.groupby("microgrid_id", sort=True)[columns[1:]]
        .sum()
        .reset_index()
        .loc[:, columns]
    )
