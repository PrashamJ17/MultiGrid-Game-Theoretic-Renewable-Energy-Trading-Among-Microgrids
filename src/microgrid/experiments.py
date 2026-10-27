"""Controlled experiments E1-E7 (blueprint section 13).

Design rules (blueprint section 13, "Controlled experiment design")
    * Each experiment varies exactly one parameter family. Every other config
      field keeps its base value; a test checks this config by config.
    * Profiles are frozen within an experiment: the same validated table,
      recorded by SHA-256 for every run. The exception is when the varied
      parameter is itself a profile-generator parameter (E4 solar scale, E5
      market size, E7 seed); then each run regenerates the profiles from the
      base config with only that parameter changed.
    * Every run goes through ``simulation.compare_all``, so all three
      mechanisms see identical states.
    * Every run's exact config is saved next to its results. The result
      tables are deterministic (byte-identical on re-run). Wall-clock
      runtimes are not reproducible, so they go to a separate
      ``runtimes.csv``.
    * Stochastic variation (E7) is reported for every seed, plus mean,
      standard deviation and range. No single seed is singled out.

Interaction
    Orchestrates ``simulation``, ``data`` and ``competitive``, and writes
    only CSV and JSON. It does no plotting: ``scripts/make_figures.py``
    draws the figures from the saved files.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .competitive import analyze_price_game
from .data import (
    generate_profiles,
    results_to_diagnostics_frame,
    results_to_settlement_frame,
    results_to_trade_frame,
    save_config,
    validate_profiles,
    write_csv,
)
from .energy import add_surplus_deficit_columns, compute_hour_states
from .metrics import METRIC_COLUMNS
from .models import CombinatorialWarning, ScenarioConfig
from .simulation import ComparisonResult, compare_all


@dataclass(frozen=True)
class ExperimentSpec:
    """One blueprint experiment: the single parameter it varies and the values it takes."""

    experiment_id: str
    title: str
    variable: str
    unit: str
    values: tuple[Any, ...]
    regenerates_profiles: bool
    question: str
    label: str

    def run_id(self, value: Any) -> str:
        text = f"{value:.2f}" if isinstance(value, float) else str(value)
        return f"{self.experiment_id}_{self.label}_{text}"


def _identity(cfg: ScenarioConfig, _: Any) -> ScenarioConfig:
    return cfg


def _spread(cfg: ScenarioConfig, spread: float) -> ScenarioConfig:
    return dataclasses.replace(cfg, retail_price_inr_per_kwh=cfg.export_price_inr_per_kwh + spread)


def _fees(cfg: ScenarioConfig, total: float) -> ScenarioConfig:
    return dataclasses.replace(cfg, seller_fee_inr_per_kwh=total / 2, buyer_fee_inr_per_kwh=total / 2)


def _solar(cfg: ScenarioConfig, scale: float) -> ScenarioConfig:
    return dataclasses.replace(cfg, generator=dataclasses.replace(cfg.generator, solar_scale=scale))


def _size(cfg: ScenarioConfig, n: int) -> ScenarioConfig:
    ids = tuple(f"MG{i}" for i in range(1, n + 1))
    overrides = tuple((mg, spec) for mg, spec in cfg.generator.microgrids if mg in ids)
    return dataclasses.replace(cfg, microgrid_ids=ids, generator=dataclasses.replace(cfg.generator, microgrids=overrides))


def _tick(cfg: ScenarioConfig, step: float) -> ScenarioConfig:
    return dataclasses.replace(cfg, price_step_inr=step)


def _seed(cfg: ScenarioConfig, seed: int) -> ScenarioConfig:
    return dataclasses.replace(cfg, seed=seed)


SPECS: dict[str, ExperimentSpec] = {
    "E1": ExperimentSpec(
        "E1", "Mechanism comparison", "scenario", "", ("default",), False,
        "How do net cost, gains, prices, volume and fairness differ between the baseline, cooperative and "
        "competitive mechanisms on the same day?", "reference",
    ),
    "E2": ExperimentSpec(
        "E2", "Tariff spread", "spread_inr_per_kwh", "INR/kWh", (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0), False,
        "How does community welfare respond to the retail-export spread b - a (export price fixed)?", "spread",
    ),
    "E3": ExperimentSpec(
        "E3", "Fee threshold", "total_fee_inr_per_kwh", "INR/kWh",
        (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5), False,
        "How do P2P volume and total gain fall as seller + buyer fees (split equally) approach the spread?",
        "fee",
    ),
    "E4": ExperimentSpec(
        "E4", "Solar scale", "solar_scale", "x", (0.5, 1.0, 1.5, 2.0), True,
        "How do surplus, traded volume and cost change when every microgrid's solar output is scaled?",
        "solar",
    ),
    "E5": ExperimentSpec(
        "E5", "Market size", "microgrids", "players", (2, 4, 6), True,
        "How do runtime, equilibrium existence and equity change with the number of microgrids?", "size",
    ),
    "E6": ExperimentSpec(
        "E6", "Strategy resolution", "price_step_inr", "INR/kWh", (1.0, 0.5, 0.25), False,
        "How sensitive are equilibrium existence, prices and runtime to the seller price tick?", "tick",
    ),
    "E7": ExperimentSpec(
        "E7", "Reproducibility across seeds", "seed", "", tuple(range(42, 52)), True,
        "How much do gains and fairness vary across ten synthetic days (seeds 42-51)?", "seed",
    ),
}

_APPLY: dict[str, Callable[[ScenarioConfig, Any], ScenarioConfig]] = {
    "E1": _identity, "E2": _spread, "E3": _fees, "E4": _solar, "E5": _size, "E6": _tick, "E7": _seed,
}

#: Metrics summarised across runs (mean / std / min / max), used for E7.
SUMMARY_METRICS = (
    "net_community_cost_inr",
    "total_gain_inr",
    "total_p2p_volume_kwh",
    "jain_fairness",
    "efficiency",
    "no_pure_ne_intervals",
)


def profiles_sha256(profiles: pd.DataFrame) -> str:
    """Fingerprint of a validated profile table (identical tables give identical hashes)."""
    return hashlib.sha256(profiles.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExperimentRun:
    run_id: str
    value: Any
    config: ScenarioConfig
    profiles_sha256: str
    comparison: ComparisonResult


@dataclass(frozen=True)
class ExperimentResult:
    """All runs of one experiment, with deterministic result tables."""

    spec: ExperimentSpec
    base_config: ScenarioConfig
    runs: tuple[ExperimentRun, ...]

    def runs_frame(self) -> pd.DataFrame:
        """One row per run and mechanism: the varied value, profile totals and every metric except runtime."""
        rows = []
        for run in self.runs:
            totals = add_surplus_deficit_columns(run.comparison.validated_profiles, tol=run.config.tol)
            for mechanism in run.comparison.mechanisms:
                metrics = run.comparison.metrics[mechanism]
                rows.append(
                    {
                        "experiment": self.spec.experiment_id,
                        "run_id": run.run_id,
                        self.spec.variable: run.value,
                        "microgrids": len(run.config.microgrid_ids),
                        "total_surplus_kwh": float(totals["surplus_kwh"].sum()),
                        "total_deficit_kwh": float(totals["deficit_kwh"].sum()),
                        "profiles_sha256": run.profiles_sha256,
                        **{k: metrics[k] for k in METRIC_COLUMNS if k not in ("runtime_seconds", "scenario_id")},
                    }
                )
        return pd.DataFrame(rows)

    def players_frame(self) -> pd.DataFrame:
        frames = [
            run.comparison.player_frame().assign(run_id=run.run_id, **{self.spec.variable: run.value})
            for run in self.runs
        ]
        frame = pd.concat(frames, ignore_index=True)
        front = ["run_id", self.spec.variable]
        return frame[front + [c for c in frame.columns if c not in front]]

    def intervals_frame(self) -> pd.DataFrame:
        """One row per run, mechanism and interval: status, volume, gain, equilibrium count and regret."""
        rows = []
        for run in self.runs:
            for mechanism, results in run.comparison.results:
                for r in results:
                    d = r.diagnostics
                    rows.append(
                        {
                            "run_id": run.run_id,
                            self.spec.variable: run.value,
                            "mechanism": mechanism,
                            "day": r.day,
                            "hour": r.hour,
                            "status": d.status,
                            "p2p_volume_kwh": r.total_p2p_volume_kwh,
                            "gain_inr": r.total_gain_inr,
                            "net_cost_inr": r.total_cost_inr,
                            "equilibrium_count": d.equilibrium_count,
                            "examined_profiles": d.examined_profiles,
                            "max_regret_inr": d.max_regret_inr,
                        }
                    )
        return pd.DataFrame(rows)

    def runtimes_frame(self) -> pd.DataFrame:
        """Wall-clock solve time per run and mechanism (not reproducible; kept out of the result tables)."""
        return pd.DataFrame(
            [
                {
                    "run_id": run.run_id,
                    self.spec.variable: run.value,
                    "microgrids": len(run.config.microgrid_ids),
                    "mechanism": mechanism,
                    "runtime_seconds": seconds,
                }
                for run in self.runs
                for mechanism, seconds in run.comparison.runtimes_seconds
            ]
        )

    def summary_frame(self) -> pd.DataFrame:
        """Mean, standard deviation (n-1), min and max of key metrics across runs, per mechanism."""
        runs = self.runs_frame()
        rows = []
        for mechanism, group in runs.groupby("mechanism", sort=False):
            for metric in SUMMARY_METRICS:
                values = pd.to_numeric(group[metric], errors="coerce").dropna()
                rows.append(
                    {
                        "mechanism": mechanism,
                        "metric": metric,
                        "n": int(values.size),
                        "mean": float(values.mean()) if values.size else None,
                        "std": float(values.std(ddof=1)) if values.size > 1 else None,
                        "min": float(values.min()) if values.size else None,
                        "max": float(values.max()) if values.size else None,
                    }
                )
        return pd.DataFrame(rows)


def payoff_curves_frame(comparison: ComparisonResult) -> pd.DataFrame:
    """For every competitive game interval: each seller's utility as its own price varies.

    The other sellers are held at the selected equilibrium, or at the NOT
    NASH diagnostic profile when no pure NE exists. This is blueprint
    figure 6.
    """
    cfg = comparison.config
    states = {(s.day, s.hour): s for s in compute_hour_states(comparison.validated_profiles, cfg.tol)}
    rows = []
    for result in comparison.results_for("competitive"):
        if not result.diagnostics.examined_profiles:
            continue
        analysis = analyze_price_game(states[(result.day, result.hour)], cfg, comparison.tariffs)
        is_nash = bool(analysis.equilibria)
        held = analysis.selected_equilibrium if is_nash else analysis.min_max_regret_profile
        for i, seller in enumerate(analysis.sellers):
            for price in analysis.price_grids[i]:
                profile = list(held)
                profile[i] = price
                rows.append(
                    {
                        "day": result.day,
                        "hour": result.hour,
                        "status": result.diagnostics.status,
                        "held_profile_is_nash": is_nash,
                        "seller": seller,
                        "held_price_inr_per_kwh": held[i],
                        "own_price_inr_per_kwh": price,
                        "utility_inr": analysis.payoffs(profile)[seller],
                    }
                )
    return pd.DataFrame(rows)


def run_experiment(experiment_id: str, base_cfg: ScenarioConfig, base_profiles: pd.DataFrame) -> ExperimentResult:
    """Run every value of one experiment through ``compare_all``.

    ``base_profiles`` is the frozen table used by the experiments that do not
    vary a generator parameter. Enumeration-cap warnings are recorded in the
    diagnostics rather than emitted.
    """
    if experiment_id not in SPECS:
        raise ValueError(f"unknown experiment {experiment_id!r}; choose from {sorted(SPECS)}")
    spec = SPECS[experiment_id]
    frozen = validate_profiles(base_profiles, base_cfg)
    runs = []
    for value in spec.values:
        run_id = spec.run_id(value)
        cfg = dataclasses.replace(_APPLY[experiment_id](base_cfg, value), scenario_id=run_id)
        profiles = generate_profiles(cfg) if spec.regenerates_profiles else frozen
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CombinatorialWarning)
            comparison = compare_all(profiles, cfg)
        runs.append(ExperimentRun(run_id, value, cfg, profiles_sha256(comparison.validated_profiles), comparison))
    return ExperimentResult(spec, base_cfg, tuple(runs))


def save_experiment(result: ExperimentResult, out_dir: str | Path) -> list[Path]:
    """Write every table, each run's exact config and an ``experiment.json`` manifest."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = result.spec
    written = [
        write_csv(result.runs_frame(), out / "runs.csv"),
        write_csv(result.players_frame(), out / "players.csv"),
        write_csv(result.intervals_frame(), out / "intervals.csv"),
        write_csv(result.summary_frame(), out / "summary.csv"),
        write_csv(result.runtimes_frame(), out / "runtimes.csv"),
    ]
    for run in result.runs:
        written.append(save_config(run.config, out / "configs" / f"{run.run_id}.json"))
    if spec.experiment_id == "E1":
        comparison = result.runs[0].comparison
        all_results = [r for _, rs in comparison.results for r in rs]
        written += [
            write_csv(comparison.validated_profiles, out / "profiles.csv"),
            write_csv(results_to_trade_frame(all_results), out / "trades.csv"),
            write_csv(results_to_settlement_frame(all_results), out / "settlement.csv"),
            write_csv(results_to_diagnostics_frame(all_results), out / "diagnostics.csv"),
            write_csv(payoff_curves_frame(comparison), out / "payoff_curves.csv"),
        ]
    manifest = {
        "experiment": spec.experiment_id,
        "title": spec.title,
        "question": spec.question,
        "variable": spec.variable,
        "unit": spec.unit,
        "values": list(spec.values),
        "regenerates_profiles": spec.regenerates_profiles,
        "runs": [
            {"run_id": r.run_id, "value": r.value, "profiles_sha256": r.profiles_sha256,
             "config": f"configs/{r.run_id}.json"}
            for r in result.runs
        ],
        "base_config": result.base_config.to_dict(),
        "environment": dict(result.runs[0].comparison.environment),
    }
    manifest_path = out / "experiment.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    written.append(manifest_path)
    return written


def run_and_save(
    experiment_ids: list[str], base_cfg: ScenarioConfig, base_profiles: pd.DataFrame, out_root: str | Path
) -> dict[str, ExperimentResult]:
    """Run and save several experiments into ``out_root/<experiment_id>/``; returns the results."""
    results = {}
    for experiment_id in experiment_ids:
        result = run_experiment(experiment_id, base_cfg, base_profiles)
        save_experiment(result, Path(out_root) / experiment_id)
        results[experiment_id] = result
    return results
