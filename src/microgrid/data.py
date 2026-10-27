"""Dataset and configuration I/O.

Responsibilities
    * Load and save scenario configs (JSON) and hourly tariff tables (CSV).
    * Validate energy-profile tables against the blueprint's dataset contract
      and return a normalized, sorted copy. The caller's frame is never modified.
    * Generate reproducible synthetic 24-hour profiles from a seed.
    * Convert market results into the canonical trade/settlement CSV tables.

Interaction
    Depends only on ``models``. ``energy`` and ``baseline`` consume the
    validated frames it returns. ``cli`` and ``app`` use it for all file access.

Canonical profile schema
    day (int >= 0), hour (int 0..23), microgrid_id (non-empty str),
    generation_kwh (finite float >= 0), demand_kwh (finite float >= 0).
    Unique key: (day, hour, microgrid_id). Rows are sorted by that key.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .models import (
    DIAGNOSTICS_CSV_COLUMNS,
    HOURS_PER_DAY,
    SETTLEMENT_CSV_COLUMNS,
    SETTLEMENT_DETAIL_COLUMNS,
    TRADE_CSV_COLUMNS,
    DataValidationError,
    MarketResult,
    MicrogridProfileSpec,
    ScenarioConfig,
    TariffSchedule,
)

PROFILE_COLUMNS = ("day", "hour", "microgrid_id", "generation_kwh", "demand_kwh")
KEY_COLUMNS = ("day", "hour", "microgrid_id")
ENERGY_COLUMNS = ("generation_kwh", "demand_kwh")
TARIFF_COLUMNS = ("hour", "retail_price_inr_per_kwh", "export_price_inr_per_kwh")

#: How many offending rows/keys an error message lists before summarising.
MAX_REPORTED = 10
#: Synthetic profiles are rounded to this many decimals (0.1 Wh) for stable CSVs.
GENERATED_DECIMALS = 4


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"file not found: {path}")
    try:
        return pd.read_csv(path, skipinitialspace=True)
    except pd.errors.EmptyDataError:
        raise DataValidationError(
            f"{path}: file is empty (no header and no rows)", [f"{path}: empty file"]
        ) from None
    except pd.errors.ParserError as exc:
        raise DataValidationError(f"{path}: malformed CSV ({exc})", [str(exc)]) from None


def write_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a table with Unix line endings so output is byte-identical across platforms."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def _row_ref(label: Any, from_csv: bool) -> str:
    """Describe a row. For CSV input, data row 0 is on line 2 (line 1 is the header)."""
    if from_csv and isinstance(label, (int, np.integer)):
        return f"row {int(label)} (CSV line {int(label) + 2})"
    return f"row {label}"


def _describe(labels: Sequence[Any], values: Sequence[Any] | None, from_csv: bool) -> str:
    labels = list(labels)
    parts = []
    for i, label in enumerate(labels[:MAX_REPORTED]):
        text = _row_ref(label, from_csv)
        if values is not None:
            text += f" = {values[i]!r}"
        parts.append(text)
    if len(labels) > MAX_REPORTED:
        parts.append(f"... and {len(labels) - MAX_REPORTED} more")
    return "; ".join(parts)


def _raise_if_problems(problems: list[str], what: str) -> None:
    if problems:
        bullet_list = "\n".join(f"  - {p}" for p in problems)
        raise DataValidationError(
            f"{what} failed validation with {len(problems)} problem(s):\n{bullet_list}", problems
        )


# ---------------------------------------------------------------------------
# Configuration and tariffs
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> ScenarioConfig:
    """Load and validate a scenario config JSON file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"{path}: invalid JSON ({exc})", [str(exc)]) from None
    try:
        return ScenarioConfig.from_dict(data)
    except DataValidationError as exc:
        raise DataValidationError(f"{path}: {exc}", exc.problems or [str(exc)]) from None


def save_config(cfg: ScenarioConfig, path: str | Path) -> Path:
    """Serialise the exact config used for a run (stored next to its results)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def load_tariffs_csv(path: str | Path) -> TariffSchedule:
    """Load an hour-of-day tariff table with columns ``hour, retail_price_inr_per_kwh,
    export_price_inr_per_kwh``. It needs exactly one row for each hour 0..23."""
    path = Path(path)
    frame = _read_csv(path)
    missing = [c for c in TARIFF_COLUMNS if c not in frame.columns]
    if missing:
        msg = f"{path}: missing tariff column(s) {missing}; required columns are {list(TARIFF_COLUMNS)}"
        raise DataValidationError(msg, [msg])

    problems: list[str] = []
    hours = pd.to_numeric(frame["hour"], errors="coerce").astype("float64")
    bad = ~np.isfinite(hours) | (np.floor(hours) != hours) | (hours < 0) | (hours >= HOURS_PER_DAY)
    if bad.any():
        problems.append(
            f"column 'hour' must hold integers 0..{HOURS_PER_DAY - 1}: "
            f"{_describe(frame.index[bad], frame['hour'][bad].tolist(), True)}"
        )
    for column in TARIFF_COLUMNS[1:]:
        prices = pd.to_numeric(frame[column], errors="coerce").astype("float64")
        invalid = ~np.isfinite(prices) | (prices < 0)
        if invalid.any():
            problems.append(
                f"column '{column}' must hold finite prices >= 0: "
                f"{_describe(frame.index[invalid], frame[column][invalid].tolist(), True)}"
            )
    _raise_if_problems(problems, f"tariff table {path}")

    hour_values = hours.astype("int64")
    duplicates = sorted(set(hour_values[hour_values.duplicated()].tolist()))
    absent = sorted(set(range(HOURS_PER_DAY)) - set(hour_values.tolist()))
    if duplicates:
        problems.append(f"duplicate tariff rows for hour(s) {duplicates}")
    if absent:
        problems.append(f"missing tariff rows for hour(s) {absent}")
    _raise_if_problems(problems, f"tariff table {path}")

    ordered = frame.assign(hour=hour_values).sort_values("hour")
    return TariffSchedule(
        retail_inr_per_kwh=tuple(float(v) for v in ordered["retail_price_inr_per_kwh"]),
        export_inr_per_kwh=tuple(float(v) for v in ordered["export_price_inr_per_kwh"]),
    )


def tariffs_to_frame(tariffs: TariffSchedule) -> pd.DataFrame:
    """Tariff schedule as a table (written next to results when a custom schedule is used)."""
    return pd.DataFrame(
        {
            "hour": range(HOURS_PER_DAY),
            "retail_price_inr_per_kwh": tariffs.retail_inr_per_kwh,
            "export_price_inr_per_kwh": tariffs.export_inr_per_kwh,
        }
    )


# ---------------------------------------------------------------------------
# Profile validation
# ---------------------------------------------------------------------------


def _check_integer_column(
    frame: pd.DataFrame,
    labels: pd.Index,
    column: str,
    low: int,
    high: int,
    problems: list[str],
    from_csv: bool,
    range_hint: str,
) -> tuple[pd.Series, pd.Series]:
    raw = frame[column]
    numeric = pd.to_numeric(raw, errors="coerce").astype("float64")
    missing = raw.isna()
    non_numeric = numeric.isna() & ~missing
    finite = pd.Series(np.isfinite(numeric.to_numpy()), index=frame.index)
    non_finite = ~finite & ~numeric.isna()
    non_integer = finite & (np.floor(numeric) != numeric)
    out_of_range = finite & ~non_integer & ((numeric < low) | (numeric > high))

    for mask, reason in (
        (missing, "missing value"),
        (non_numeric | non_finite, "non-numeric or non-finite value"),
        (non_integer, "non-integer value"),
        (out_of_range, f"value outside {range_hint}"),
    ):
        if mask.any():
            problems.append(
                f"column '{column}': {reason} at "
                f"{_describe(labels[mask.to_numpy()], raw[mask].tolist(), from_csv)}"
            )
    valid = finite & ~non_integer & ~out_of_range
    return numeric, valid


def validate_profiles(
    profiles: pd.DataFrame,
    cfg: ScenarioConfig,
    *,
    require_full_horizon: bool | None = None,
    source: str | None = None,
) -> pd.DataFrame:
    """Validate an energy-profile table and return a normalized, sorted copy.

    Every problem found is collected and reported together in one
    ``DataValidationError`` (a ``ValueError``) that names the row and the column.
    The input frame is not modified.

    Checks, in order:
        1. all required columns are present (the exact missing names are reported)
        2. at least one data row
        3. ``microgrid_id``: non-empty after stripping whitespace, and a configured id
        4. ``day``: integer in 0..days-1; ``hour``: integer in 0..23
        5. energy columns: numeric, finite and >= 0
        6. no duplicate (day, hour, microgrid_id) keys
        7. every (day, hour) present contains every configured microgrid
        8. if ``require_full_horizon`` (default: the config's setting), every
           day x hour slot of the configured horizon is present

    Extra columns are dropped from the returned frame.

    ``source`` is the file path the frame was read from. When it is given,
    errors also report CSV line numbers (data row ``i`` is on line ``i + 2``).
    """
    if not isinstance(profiles, pd.DataFrame):
        raise TypeError(f"profiles must be a pandas DataFrame, got {type(profiles).__name__}")
    what = f"profile table {source}" if source else "profile table"
    from_csv = source is not None
    full_horizon = cfg.require_full_horizon if require_full_horizon is None else require_full_horizon

    missing_columns = [c for c in PROFILE_COLUMNS if c not in profiles.columns]
    if missing_columns:
        msg = (
            f"{what} is missing required column(s) {missing_columns}; "
            f"required columns are {list(PROFILE_COLUMNS)}"
        )
        raise DataValidationError(msg, [msg])
    if len(profiles) == 0:
        msg = f"{what} has no data rows"
        raise DataValidationError(msg, [msg])

    labels = profiles.index
    frame = profiles.loc[:, list(PROFILE_COLUMNS)].reset_index(drop=True)
    problems: list[str] = []

    # 3. microgrid ids
    ids = frame["microgrid_id"].astype("string").str.strip().fillna("").astype(object)
    blank = ids.eq("")
    if blank.any():
        problems.append(
            "column 'microgrid_id': missing or empty id at "
            f"{_describe(labels[blank.to_numpy()], None, from_csv)}"
        )
    unknown = ~blank & ~ids.isin(set(cfg.microgrid_ids))
    if unknown.any():
        problems.append(
            f"column 'microgrid_id': unknown id(s) {sorted(set(ids[unknown]))} "
            f"(configured ids: {list(cfg.microgrid_ids)}) at "
            f"{_describe(labels[unknown.to_numpy()], ids[unknown].tolist(), from_csv)}"
        )

    # 4. time keys
    day, day_ok = _check_integer_column(
        frame, labels, "day", 0, cfg.days - 1, problems, from_csv,
        range_hint=f"0..{cfg.days - 1} (config days={cfg.days})",
    )
    hour, hour_ok = _check_integer_column(
        frame, labels, "hour", 0, HOURS_PER_DAY - 1, problems, from_csv,
        range_hint=f"0..{HOURS_PER_DAY - 1}",
    )

    # 5. energy values
    energy: dict[str, pd.Series] = {}
    for column in ENERGY_COLUMNS:
        raw = frame[column]
        numeric = pd.to_numeric(raw, errors="coerce").astype("float64")
        missing = raw.isna()
        non_numeric = numeric.isna() & ~missing
        infinite = pd.Series(np.isinf(numeric.to_numpy()), index=frame.index)
        negative = (numeric < 0) & ~infinite
        for mask, reason in (
            (missing, "missing value (NaN/empty)"),
            (non_numeric, "non-numeric value"),
            (infinite, "non-finite value"),
            (negative, "negative energy (must be >= 0 kWh)"),
        ):
            if mask.any():
                problems.append(
                    f"column '{column}': {reason} at "
                    f"{_describe(labels[mask.to_numpy()], raw[mask].tolist(), from_csv)}"
                )
        energy[column] = numeric

    # 6. duplicate keys (only among rows whose keys are themselves valid)
    key_ok = day_ok & hour_ok & ~blank & ~unknown
    keys = pd.DataFrame({"day": day, "hour": hour, "microgrid_id": ids})
    valid_keys = keys[key_ok]
    duplicated = valid_keys[valid_keys.duplicated(keep=False)]
    if not duplicated.empty:
        groups = list(duplicated.groupby(list(KEY_COLUMNS), sort=True))
        for (d, h, mg), group in groups[:MAX_REPORTED]:
            problems.append(
                f"duplicate record for (day={int(d)}, hour={int(h)}, microgrid_id={mg!r}) at "
                f"{_describe(labels[group.index.to_numpy()], None, from_csv)}"
            )
        if len(groups) > MAX_REPORTED:
            problems.append(f"... and {len(groups) - MAX_REPORTED} more duplicate keys")

    _raise_if_problems(problems, what)

    clean = pd.DataFrame(
        {
            "day": day.astype("int64"),
            "hour": hour.astype("int64"),
            "microgrid_id": ids,
            "generation_kwh": energy["generation_kwh"] + 0.0,  # + 0.0 turns -0.0 into 0.0
            "demand_kwh": energy["demand_kwh"] + 0.0,
        }
    )

    # 7-8. completeness: missing records
    present = set(zip(clean["day"].tolist(), clean["hour"].tolist(), clean["microgrid_id"].tolist()))
    if full_horizon:
        slots = [(d, h) for d in range(cfg.days) for h in range(HOURS_PER_DAY)]
    else:
        slots = sorted(set(zip(clean["day"].tolist(), clean["hour"].tolist())))
    expected = [(d, h, mg) for d, h in slots for mg in sorted(cfg.microgrid_ids)]
    absent = [key for key in expected if key not in present]
    if absent:
        shown = ", ".join(f"(day={d}, hour={h}, microgrid_id={mg!r})" for d, h, mg in absent[:MAX_REPORTED])
        more = f" ... and {len(absent) - MAX_REPORTED} more" if len(absent) > MAX_REPORTED else ""
        scope = (
            f"the full horizon needs {cfg.expected_rows} rows "
            f"({cfg.days} day(s) x {HOURS_PER_DAY} hours x {len(cfg.microgrid_ids)} microgrids), "
            f"found {len(clean)}"
            if full_horizon
            else "every (day, hour) present must contain all configured microgrids"
        )
        problems.append(f"missing {len(absent)} record(s): {shown}{more}; {scope}")
    _raise_if_problems(problems, what)

    return clean.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(drop=True)


def load_and_validate_profiles(
    path: str | Path,
    cfg: ScenarioConfig,
    *,
    require_full_horizon: bool | None = None,
) -> pd.DataFrame:
    """Read a profile CSV and validate it (see ``validate_profiles``)."""
    path = Path(path)
    frame = _read_csv(path)
    return validate_profiles(
        frame, cfg, require_full_horizon=require_full_horizon, source=str(path)
    )


def save_profiles(profiles: pd.DataFrame, path: str | Path) -> Path:
    """Write the canonical profile columns to CSV."""
    return write_csv(profiles.loc[:, list(PROFILE_COLUMNS)], path)


# ---------------------------------------------------------------------------
# Synthetic profile generator
# ---------------------------------------------------------------------------

#: Hand-designed, illustrative hourly demand shapes for hours 0..23 (peak = 1.0).
#: They are not measured data.
DEMAND_TEMPLATES: dict[str, tuple[float, ...]] = {
    # morning and evening peaks, low midday occupancy
    "residential": (
        0.35, 0.30, 0.28, 0.27, 0.28, 0.35, 0.55, 0.80, 0.75, 0.55, 0.45, 0.42,
        0.40, 0.40, 0.42, 0.48, 0.60, 0.80, 0.95, 1.00, 0.95, 0.80, 0.60, 0.45,
    ),
    # teaching day 9-17, quiet nights
    "campus": (
        0.25, 0.22, 0.20, 0.20, 0.20, 0.22, 0.30, 0.50, 0.80, 0.95, 1.00, 1.00,
        0.95, 1.00, 1.00, 0.95, 0.85, 0.65, 0.50, 0.45, 0.40, 0.35, 0.30, 0.27,
    ),
    # business hours 9-20
    "commercial": (
        0.20, 0.18, 0.18, 0.18, 0.18, 0.20, 0.25, 0.40, 0.65, 0.85, 0.95, 1.00,
        1.00, 1.00, 0.98, 0.95, 0.92, 0.90, 0.85, 0.80, 0.60, 0.40, 0.28, 0.22,
    ),
    # high base load with day-shift ramp
    "industrial": (
        0.70, 0.70, 0.68, 0.68, 0.70, 0.75, 0.85, 0.95, 1.00, 1.00, 1.00, 0.98,
        0.95, 0.98, 1.00, 1.00, 0.98, 0.95, 0.90, 0.85, 0.80, 0.78, 0.75, 0.72,
    ),
}

#: Default synthetic parameters, assigned by a microgrid's position in
#: ``cfg.microgrid_ids`` (cycled for more than six players). The first four
#: give two midday sellers (MG1, MG3) and two midday buyers (MG2, MG4).
DEFAULT_PROFILE_SPECS: tuple[MicrogridProfileSpec, ...] = (
    MicrogridProfileSpec("residential", solar_peak_kwh=45.0, demand_peak_kwh=25.0),
    MicrogridProfileSpec("commercial", solar_peak_kwh=12.0, demand_peak_kwh=28.0),
    MicrogridProfileSpec("campus", solar_peak_kwh=30.0, demand_peak_kwh=18.0),
    MicrogridProfileSpec("industrial", solar_peak_kwh=6.0, demand_peak_kwh=32.0),
    MicrogridProfileSpec("residential", solar_peak_kwh=20.0, demand_peak_kwh=22.0),
    MicrogridProfileSpec("commercial", solar_peak_kwh=35.0, demand_peak_kwh=24.0),
)


def solar_shape(sunrise_hour: float = 6.0, sunset_hour: float = 18.0) -> np.ndarray:
    """Normalized hourly solar energy shape (peak hour = 1.0, night = exactly 0).

    It is a half-sine between sunrise and sunset, sampled at each hour's
    midpoint (h + 0.5). The value is the hour's energy relative to the
    sunniest hour.
    """
    midpoints = np.arange(HOURS_PER_DAY, dtype=float) + 0.5
    phase = (midpoints - sunrise_hour) / (sunset_hour - sunrise_hour)
    shape = np.where((phase > 0.0) & (phase < 1.0), np.sin(np.pi * phase), 0.0)
    peak = shape.max()
    return shape / peak if peak > 0 else shape


def profile_spec_for(cfg: ScenarioConfig, position: int, microgrid_id: str) -> MicrogridProfileSpec:
    """Explicit override from the config if present, else the positional default."""
    override = cfg.generator.spec_for(microgrid_id)
    if override is not None:
        return override
    return DEFAULT_PROFILE_SPECS[position % len(DEFAULT_PROFILE_SPECS)]


def generate_profiles(cfg: ScenarioConfig) -> pd.DataFrame:
    """Generate reproducible synthetic hourly profiles (kWh per hour).

    generation[d,h,i] = solar_peak_i * solar_scale * solar_shape[h] * max(0, 1 + e_g)
    demand[d,h,i]     = demand_peak_i * demand_scale * template_i[h] * max(0, 1 + e_d)
    with e ~ Normal(0, noise_std_fraction).

    Randomness comes only from ``np.random.default_rng(cfg.seed)``; the global
    NumPy state is never used. Draw order is fixed: all generation noise
    (shape days x 24 x N), then all demand noise. Values are rounded to
    ``GENERATED_DECIMALS``. Because the noise is multiplicative and the
    scales are applied last, changing ``solar_scale``/``demand_scale``
    scales the profiles without changing their shape.
    """
    gen = cfg.generator
    ids = cfg.microgrid_ids
    rng = np.random.default_rng(cfg.seed)
    size = (cfg.days, HOURS_PER_DAY, len(ids))
    generation_noise = rng.normal(0.0, gen.noise_std_fraction, size=size)
    demand_noise = rng.normal(0.0, gen.noise_std_fraction, size=size)

    specs = [profile_spec_for(cfg, k, mg) for k, mg in enumerate(ids)]
    solar = solar_shape(gen.sunrise_hour, gen.sunset_hour)  # (24,)
    solar_peaks = np.array([s.solar_peak_kwh for s in specs]) * gen.solar_scale  # (N,)
    demand_shapes = np.array([DEMAND_TEMPLATES[s.demand_template] for s in specs]).T  # (24, N)
    demand_peaks = np.array([s.demand_peak_kwh for s in specs]) * gen.demand_scale  # (N,)

    generation = solar[None, :, None] * solar_peaks[None, None, :] * np.maximum(1.0 + generation_noise, 0.0)
    demand = demand_shapes[None, :, :] * demand_peaks[None, None, :] * np.maximum(1.0 + demand_noise, 0.0)
    generation = np.round(generation, GENERATED_DECIMALS) + 0.0
    demand = np.round(demand, GENERATED_DECIMALS) + 0.0

    day_idx, hour_idx, mg_idx = np.meshgrid(
        np.arange(cfg.days), np.arange(HOURS_PER_DAY), np.arange(len(ids)), indexing="ij"
    )
    frame = pd.DataFrame(
        {
            "day": day_idx.ravel(),
            "hour": hour_idx.ravel(),
            "microgrid_id": np.array(ids, dtype=object)[mg_idx.ravel()],
            "generation_kwh": generation.ravel(),
            "demand_kwh": demand.ravel(),
        }
    )
    return validate_profiles(frame, cfg, require_full_horizon=True)


# ---------------------------------------------------------------------------
# Output tables
# ---------------------------------------------------------------------------


def results_to_settlement_frame(
    results: Iterable[MarketResult], *, detailed: bool = False
) -> pd.DataFrame:
    """Per-player settlement rows in the blueprint's CSV schema.

    ``detailed=True`` also adds the money components (grid purchase, export
    revenue, P2P payments, fees).
    """
    columns = list(SETTLEMENT_CSV_COLUMNS + (SETTLEMENT_DETAIL_COLUMNS if detailed else ()))
    records = [s.to_record(detailed) for r in results for s in r.settlements]
    return pd.DataFrame.from_records(records, columns=columns)


def results_to_trade_frame(results: Iterable[MarketResult]) -> pd.DataFrame:
    """Trade log in the blueprint's CSV schema (empty for the baseline)."""
    records = [
        {
            "scenario_id": r.scenario_id,
            "mechanism": r.mechanism,
            "day": t.day,
            "hour": t.hour,
            "seller_id": t.seller_id,
            "buyer_id": t.buyer_id,
            "quantity_kwh": t.quantity_kwh,
            "price_inr_per_kwh": t.price_inr_per_kwh,
            "seller_fee_inr_per_kwh": t.seller_fee_inr_per_kwh,
            "buyer_fee_inr_per_kwh": t.buyer_fee_inr_per_kwh,
        }
        for r in results
        for t in r.trades
    ]
    return pd.DataFrame.from_records(records, columns=list(TRADE_CSV_COLUMNS))


def diagnostics_records(results: Iterable[MarketResult]) -> list[dict[str, Any]]:
    """Blueprint section 5 solver-diagnostics records (JSON-ready; price profiles as {seller: price})."""
    records = []
    for r in results:
        d = r.diagnostics
        records.append(
            {
                "scenario_id": r.scenario_id,
                "mechanism": r.mechanism,
                "day": r.day,
                "hour": r.hour,
                "status": d.status,
                "pure_ne_found": d.pure_ne_found,
                "equilibrium_count": d.equilibrium_count,
                "selected_profile": None if d.selected_profile is None else dict(d.selected_profile),
                "max_unilateral_regret_inr": d.max_regret_inr,
                "examined_profiles": d.examined_profiles,
                "not_nash_profile": None if d.not_nash_profile is None else dict(d.not_nash_profile),
                "optimal_welfare_inr": d.optimal_welfare_inr,
                "p2p_volume_kwh": r.total_p2p_volume_kwh,
                "message": d.message,
            }
        )
    return records


def results_to_diagnostics_frame(results: Iterable[MarketResult]) -> pd.DataFrame:
    """Diagnostics as a CSV-ready table; price profiles are JSON strings (empty when absent)."""
    records = diagnostics_records(results)
    for record in records:
        for key in ("selected_profile", "not_nash_profile"):
            record[key] = "" if record[key] is None else json.dumps(record[key])
    return pd.DataFrame.from_records(records, columns=list(DIAGNOSTICS_CSV_COLUMNS))
