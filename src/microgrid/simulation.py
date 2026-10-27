"""Simulation controller: run every mechanism on the same frozen scenario and compare them.

``compare_all`` validates the profiles once, builds the immutable
``HourState`` objects once, and runs the baseline, cooperative and
competitive mechanisms on those identical states, timing each one. It checks
that the caller's input frame is unchanged after every mechanism (T16) and
computes the metrics with the cooperative LP optimum as the efficiency
reference.

Interaction: orchestrates ``baseline``, ``cooperative`` and ``competitive``;
uses ``metrics`` for the summaries and ``data`` / ``settlement`` for the
tables. It contains no market maths of its own.
"""

from __future__ import annotations

import math
import platform
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
import scipy

from . import __version__
from .baseline import solve_baseline
from .competitive import solve_competitive
from .cooperative import solve_cooperative
from .data import (
    results_to_diagnostics_frame,
    results_to_settlement_frame,
    results_to_trade_frame,
    validate_profiles,
)
from .energy import compute_hour_states
from .metrics import METRIC_COLUMNS, compute_metrics
from .models import MarketResult, ScenarioConfig, TariffSchedule
from .settlement import summarize_settlements

MECHANISMS = ("baseline", "cooperative", "competitive")
_SOLVERS = {"baseline": solve_baseline, "cooperative": solve_cooperative, "competitive": solve_competitive}


@dataclass(frozen=True)
class ComparisonResult:
    """All mechanisms' results on one frozen scenario, with metrics, runtimes and environment.

    ``validated_profiles`` and ``metrics`` are shared objects: treat them as read-only.
    """

    config: ScenarioConfig
    tariffs: TariffSchedule
    results: tuple[tuple[str, tuple[MarketResult, ...]], ...]
    runtimes_seconds: tuple[tuple[str, float], ...]
    environment: tuple[tuple[str, str], ...]
    metrics: dict[str, dict[str, Any]] = field(compare=False)
    validated_profiles: pd.DataFrame = field(compare=False, repr=False)

    @property
    def mechanisms(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.results)

    def results_for(self, mechanism: str) -> tuple[MarketResult, ...]:
        for name, results in self.results:
            if name == mechanism:
                return results
        raise KeyError(f"mechanism {mechanism!r} was not run; available: {self.mechanisms}")

    def _all_results(self) -> list[MarketResult]:
        return [r for _, results in self.results for r in results]

    def summary_frame(self) -> pd.DataFrame:
        """One row per mechanism with every scalar metric (None = N/A)."""
        rows = [{k: self.metrics[m][k] for k in METRIC_COLUMNS} for m in self.mechanisms]
        return pd.DataFrame(rows, columns=list(METRIC_COLUMNS))

    def player_frame(self) -> pd.DataFrame:
        """Per mechanism and microgrid: P2P flows, grid flows, baseline cost, net cost and gain."""
        frames = [summarize_settlements(results).assign(mechanism=m) for m, results in self.results]
        frame = pd.concat(frames, ignore_index=True)
        return frame[["mechanism"] + [c for c in frame.columns if c != "mechanism"]]

    def trades_frame(self) -> pd.DataFrame:
        return results_to_trade_frame(self._all_results())

    def settlement_frame(self) -> pd.DataFrame:
        return results_to_settlement_frame(self._all_results())

    def diagnostics_frame(self) -> pd.DataFrame:
        return results_to_diagnostics_frame(self._all_results())


def _environment() -> tuple[tuple[str, str], ...]:
    return (
        ("microgrid", __version__),
        ("python", platform.python_version()),
        ("platform", platform.platform()),
        ("numpy", np.__version__),
        ("pandas", pd.__version__),
        ("scipy", scipy.__version__),
    )


def compare_all(
    profiles: pd.DataFrame,
    cfg: ScenarioConfig,
    tariffs: TariffSchedule | None = None,
    mechanisms: Sequence[str] = MECHANISMS,
) -> ComparisonResult:
    """Run the chosen mechanisms on identical frozen states and return a ``ComparisonResult``."""
    unknown = sorted(set(mechanisms) - set(_SOLVERS))
    if unknown:
        raise ValueError(f"unknown mechanism(s) {unknown}; choose from {list(MECHANISMS)}")
    snapshot = profiles.copy(deep=True)
    validated = validate_profiles(profiles, cfg)
    schedule = tariffs if tariffs is not None else TariffSchedule.from_config(cfg)
    states = tuple(compute_hour_states(validated, cfg.tol))

    results: list[tuple[str, tuple[MarketResult, ...]]] = []
    runtimes: list[tuple[str, float]] = []
    for name in (m for m in MECHANISMS if m in mechanisms):
        start = time.perf_counter()
        outcome = tuple(_SOLVERS[name](state, cfg, schedule) for state in states)
        runtimes.append((name, time.perf_counter() - start))
        results.append((name, outcome))
        if not profiles.equals(snapshot):
            raise RuntimeError(f"the {name} mechanism modified the input profiles")

    by_name = dict(results)
    lp_optimum = (
        math.fsum(r.diagnostics.optimal_welfare_inr or 0.0 for r in by_name["cooperative"])
        if "cooperative" in by_name
        else None
    )
    metrics = {
        name: compute_metrics(outcome, lp_optimal_gain_inr=lp_optimum, runtime_seconds=dict(runtimes)[name], tol=cfg.tol)
        for name, outcome in results
    }
    return ComparisonResult(
        config=cfg,
        tariffs=schedule,
        results=tuple(results),
        runtimes_seconds=tuple(runtimes),
        environment=_environment(),
        metrics=metrics,
        validated_profiles=validated,
    )
