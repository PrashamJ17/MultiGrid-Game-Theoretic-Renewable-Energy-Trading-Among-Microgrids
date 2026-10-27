"""Typed, immutable data models shared by every MultiGrid module.

This is the bottom layer of the package. It imports nothing from the rest of
``microgrid``, performs no file I/O and does not depend on pandas, so every
other module can use it without import cycles.

Units
    Energy quantities are kWh per hourly interval. Money is INR. Unit prices
    and fees are INR/kWh.

Sign convention for costs
    Positive = the microgrid pays (net expense).
    Negative = the microgrid receives money (net income).
    Signed values are never replaced by absolute values before aggregation.
"""

from __future__ import annotations

import math
import numbers
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

HOURS_PER_DAY = 24
MECHANISMS = ("baseline", "cooperative", "competitive")
DEMAND_TEMPLATE_NAMES = ("residential", "campus", "commercial", "industrial")


# ---------------------------------------------------------------------------
# Errors and warnings
# ---------------------------------------------------------------------------


class DataValidationError(ValueError):
    """Input data or configuration violates the dataset/config contract.

    ``problems`` lists every individual violation that was found, so callers
    (and tests) can inspect them without parsing the message text.
    """

    def __init__(self, message: str, problems: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.problems: tuple[str, ...] = tuple(problems)


class EnergyBalanceError(ValueError):
    """A computed quantity breaks energy conservation or non-negativity."""


class EconomicWarning(UserWarning):
    """A configuration is allowed but economically unusual (e.g. retail < export)."""


class SettlementError(ValueError):
    """Trades or settlements are inconsistent: an unknown or ineligible trader,
    a phantom trade, broken payment cancellation, fee bookkeeping, or a
    voluntary-participation violation."""


class CombinatorialWarning(UserWarning):
    """Exact equilibrium enumeration was skipped because the profile space exceeds the configured cap."""


class SolverError(RuntimeError):
    """An optimisation solver failed or returned an unverifiable solution.

    It is raised instead of returning any result, so a failed solve can never
    reach the reporting layer.
    """


# ---------------------------------------------------------------------------
# Small validation helpers
# ---------------------------------------------------------------------------


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise DataValidationError(f"{name} must be a real number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise DataValidationError(f"{name} must be finite, got {value!r}")
    return result


def _non_negative(name: str, value: Any) -> float:
    result = _finite(name, value)
    if result < 0:
        raise DataValidationError(f"{name} must be >= 0, got {result}")
    return result


def _integer(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool):
        raise DataValidationError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, numbers.Integral):
        result = int(value)
    elif isinstance(value, numbers.Real) and float(value).is_integer():
        result = int(value)
    else:
        raise DataValidationError(f"{name} must be an integer, got {value!r}")
    if result < minimum:
        raise DataValidationError(f"{name} must be >= {minimum}, got {result}")
    return result


def _non_empty_str(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"{name} must be a non-empty string, got {value!r}")
    return value.strip()


# ---------------------------------------------------------------------------
# Scenario configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MicrogridProfileSpec:
    """Synthetic-profile parameters for one microgrid.

    ``solar_peak_kwh`` is the generation in the sunniest hour (kWh per hour).
    ``demand_peak_kwh`` is the demand in the template's peak hour (kWh per hour).
    """

    demand_template: str
    solar_peak_kwh: float
    demand_peak_kwh: float

    def __post_init__(self) -> None:
        if self.demand_template not in DEMAND_TEMPLATE_NAMES:
            raise DataValidationError(
                f"demand_template must be one of {list(DEMAND_TEMPLATE_NAMES)}, "
                f"got {self.demand_template!r}"
            )
        object.__setattr__(
            self, "solar_peak_kwh", _non_negative("solar_peak_kwh", self.solar_peak_kwh)
        )
        object.__setattr__(
            self, "demand_peak_kwh", _non_negative("demand_peak_kwh", self.demand_peak_kwh)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "demand_template": self.demand_template,
            "solar_peak_kwh": self.solar_peak_kwh,
            "demand_peak_kwh": self.demand_peak_kwh,
        }


_GENERATOR_KEYS = (
    "solar_scale",
    "demand_scale",
    "noise_std_fraction",
    "sunrise_hour",
    "sunset_hour",
    "microgrids",
)


@dataclass(frozen=True)
class GeneratorConfig:
    """Parameters of the reproducible synthetic profile generator.

    ``microgrids`` holds explicit per-microgrid overrides as sorted
    ``(microgrid_id, MicrogridProfileSpec)`` pairs. Microgrids without an
    override use the package defaults (see ``data.DEFAULT_PROFILE_SPECS``).
    """

    solar_scale: float = 1.0
    demand_scale: float = 1.0
    noise_std_fraction: float = 0.05
    sunrise_hour: float = 6.0
    sunset_hour: float = 18.0
    microgrids: tuple[tuple[str, MicrogridProfileSpec], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "solar_scale", _non_negative("solar_scale", self.solar_scale))
        object.__setattr__(
            self, "demand_scale", _non_negative("demand_scale", self.demand_scale)
        )
        noise = _non_negative("noise_std_fraction", self.noise_std_fraction)
        if noise >= 0.5:
            raise DataValidationError(
                f"noise_std_fraction must be < 0.5 to keep profiles physically plausible, got {noise}"
            )
        object.__setattr__(self, "noise_std_fraction", noise)
        sunrise = _finite("sunrise_hour", self.sunrise_hour)
        sunset = _finite("sunset_hour", self.sunset_hour)
        if not 0.0 <= sunrise < sunset <= HOURS_PER_DAY:
            raise DataValidationError(
                f"need 0 <= sunrise_hour < sunset_hour <= {HOURS_PER_DAY}, "
                f"got sunrise={sunrise}, sunset={sunset}"
            )
        object.__setattr__(self, "sunrise_hour", sunrise)
        object.__setattr__(self, "sunset_hour", sunset)

        pairs = []
        for item in self.microgrids:
            mg_id, spec = item
            if not isinstance(spec, MicrogridProfileSpec):
                raise DataValidationError(f"generator override for {mg_id!r} is not a MicrogridProfileSpec")
            pairs.append((_non_empty_str("generator microgrid id", mg_id), spec))
        ids = [mg_id for mg_id, _ in pairs]
        if len(set(ids)) != len(ids):
            raise DataValidationError(f"duplicate generator overrides: {ids}")
        object.__setattr__(self, "microgrids", tuple(sorted(pairs, key=lambda p: p[0])))

    def spec_for(self, microgrid_id: str) -> MicrogridProfileSpec | None:
        """Return the explicit override for ``microgrid_id``, or ``None``."""
        for mg_id, spec in self.microgrids:
            if mg_id == microgrid_id:
                return spec
        return None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GeneratorConfig:
        if not isinstance(data, Mapping):
            raise DataValidationError(f"'generator' must be a JSON object, got {type(data).__name__}")
        unknown = sorted(set(data) - set(_GENERATOR_KEYS))
        if unknown:
            raise DataValidationError(
                f"unknown generator key(s): {unknown}; allowed keys: {list(_GENERATOR_KEYS)}"
            )
        overrides_raw = data.get("microgrids", {}) or {}
        if not isinstance(overrides_raw, Mapping):
            raise DataValidationError("generator.microgrids must be an object keyed by microgrid id")
        overrides = []
        for mg_id, spec in overrides_raw.items():
            if not isinstance(spec, Mapping):
                raise DataValidationError(f"generator.microgrids[{mg_id!r}] must be an object")
            expected = {"demand_template", "solar_peak_kwh", "demand_peak_kwh"}
            if set(spec) != expected:
                raise DataValidationError(
                    f"generator.microgrids[{mg_id!r}] must have exactly the keys {sorted(expected)}, "
                    f"got {sorted(spec)}"
                )
            overrides.append((mg_id, MicrogridProfileSpec(**spec)))
        kwargs = {k: data[k] for k in _GENERATOR_KEYS if k in data and k != "microgrids"}
        return cls(microgrids=tuple(overrides), **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "solar_scale": self.solar_scale,
            "demand_scale": self.demand_scale,
            "noise_std_fraction": self.noise_std_fraction,
            "sunrise_hour": self.sunrise_hour,
            "sunset_hour": self.sunset_hour,
            "microgrids": {mg_id: spec.to_dict() for mg_id, spec in self.microgrids},
        }


_CONFIG_KEYS = (
    "scenario_id",
    "seed",
    "days",
    "hours_per_day",
    "microgrid_ids",
    "retail_price_inr_per_kwh",
    "export_price_inr_per_kwh",
    "seller_fee_inr_per_kwh",
    "buyer_fee_inr_per_kwh",
    "price_step_inr",
    "bargaining_seller_weight",
    "tol",
    "max_competitive_sellers",
    "require_full_horizon",
    "generator",
)


@dataclass(frozen=True)
class ScenarioConfig:
    """One reproducible scenario (blueprint section 5).

    All fields are validated on construction, so an invalid configuration
    can never reach a market solver. Fields that only later phases use
    (``price_step_inr``, ``bargaining_seller_weight``,
    ``max_competitive_sellers``, fees) are validated now so the config
    contract is frozen early.
    """

    scenario_id: str = "four_microgrids_default"
    seed: int = 42
    days: int = 1
    hours_per_day: int = HOURS_PER_DAY
    microgrid_ids: tuple[str, ...] = ("MG1", "MG2", "MG3", "MG4")
    retail_price_inr_per_kwh: float = 8.0
    export_price_inr_per_kwh: float = 3.0
    seller_fee_inr_per_kwh: float = 0.0
    buyer_fee_inr_per_kwh: float = 0.0
    price_step_inr: float = 0.5
    bargaining_seller_weight: float = 0.5
    tol: float = 1e-8
    max_competitive_sellers: int = 3
    require_full_horizon: bool = True
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        set_(self, "scenario_id", _non_empty_str("scenario_id", self.scenario_id))
        set_(self, "seed", _integer("seed", self.seed, minimum=0))
        set_(self, "days", _integer("days", self.days, minimum=1))
        hours = _integer("hours_per_day", self.hours_per_day, minimum=1)
        if hours != HOURS_PER_DAY:
            raise DataValidationError(
                f"hours_per_day must be {HOURS_PER_DAY} for the MVP (hourly intervals), got {hours}"
            )
        set_(self, "hours_per_day", hours)

        if isinstance(self.microgrid_ids, str) or not isinstance(self.microgrid_ids, Sequence):
            raise DataValidationError("microgrid_ids must be a list of strings")
        ids = tuple(_non_empty_str("microgrid id", mg) for mg in self.microgrid_ids)
        if not ids:
            raise DataValidationError("microgrid_ids must contain at least one microgrid")
        duplicates = sorted({mg for mg in ids if ids.count(mg) > 1})
        if duplicates:
            raise DataValidationError(f"microgrid_ids contains duplicates: {duplicates}")
        set_(self, "microgrid_ids", ids)

        for name in (
            "retail_price_inr_per_kwh",
            "export_price_inr_per_kwh",
            "seller_fee_inr_per_kwh",
            "buyer_fee_inr_per_kwh",
        ):
            set_(self, name, _non_negative(name, getattr(self, name)))
        if self.retail_price_inr_per_kwh < self.export_price_inr_per_kwh:
            warnings.warn(
                "retail price is below export price; P2P trade has no positive surplus "
                "and the economics are unusual",
                EconomicWarning,
                stacklevel=3,
            )

        step = _finite("price_step_inr", self.price_step_inr)
        if step <= 0:
            raise DataValidationError(f"price_step_inr must be > 0, got {step}")
        set_(self, "price_step_inr", step)

        alpha = _finite("bargaining_seller_weight", self.bargaining_seller_weight)
        if not 0.0 < alpha < 1.0:
            raise DataValidationError(
                f"bargaining_seller_weight (alpha) must be strictly between 0 and 1, got {alpha}"
            )
        set_(self, "bargaining_seller_weight", alpha)

        tol = _finite("tol", self.tol)
        if not 0.0 < tol < 1e-2:
            raise DataValidationError(f"tol must satisfy 0 < tol < 1e-2, got {tol}")
        set_(self, "tol", tol)

        set_(
            self,
            "max_competitive_sellers",
            _integer("max_competitive_sellers", self.max_competitive_sellers, minimum=1),
        )
        if not isinstance(self.require_full_horizon, bool):
            raise DataValidationError(
                f"require_full_horizon must be true or false, got {self.require_full_horizon!r}"
            )

        generator = self.generator
        if isinstance(generator, Mapping):
            generator = GeneratorConfig.from_dict(generator)
        if not isinstance(generator, GeneratorConfig):
            raise DataValidationError("generator must be a GeneratorConfig or a JSON object")
        unknown_overrides = sorted(mg for mg, _ in generator.microgrids if mg not in ids)
        if unknown_overrides:
            raise DataValidationError(
                f"generator overrides refer to unknown microgrid ids: {unknown_overrides}"
            )
        set_(self, "generator", generator)

    @property
    def expected_rows(self) -> int:
        """Rows in a complete profile table: days x hours x microgrids (96 for the MVP)."""
        return self.days * self.hours_per_day * len(self.microgrid_ids)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScenarioConfig:
        """Build a config from a JSON-like mapping.

        Unknown keys are rejected so a typo cannot silently fall back to a default.
        Missing keys take the documented defaults above.
        """
        if not isinstance(data, Mapping):
            raise DataValidationError(f"config must be a JSON object, got {type(data).__name__}")
        unknown = sorted(set(data) - set(_CONFIG_KEYS))
        if unknown:
            raise DataValidationError(
                f"unknown config key(s): {unknown}; allowed keys: {list(_CONFIG_KEYS)}"
            )
        kwargs = dict(data)
        if "microgrid_ids" in kwargs and isinstance(kwargs["microgrid_ids"], list):
            kwargs["microgrid_ids"] = tuple(kwargs["microgrid_ids"])
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form, written next to results for reproducibility."""
        return {
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "days": self.days,
            "hours_per_day": self.hours_per_day,
            "microgrid_ids": list(self.microgrid_ids),
            "retail_price_inr_per_kwh": self.retail_price_inr_per_kwh,
            "export_price_inr_per_kwh": self.export_price_inr_per_kwh,
            "seller_fee_inr_per_kwh": self.seller_fee_inr_per_kwh,
            "buyer_fee_inr_per_kwh": self.buyer_fee_inr_per_kwh,
            "price_step_inr": self.price_step_inr,
            "bargaining_seller_weight": self.bargaining_seller_weight,
            "tol": self.tol,
            "max_competitive_sellers": self.max_competitive_sellers,
            "require_full_horizon": self.require_full_horizon,
            "generator": self.generator.to_dict(),
        }


# ---------------------------------------------------------------------------
# Tariffs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TariffSchedule:
    """Hour-of-day utility tariffs, kept separate from every market algorithm.

    ``retail_inr_per_kwh[h]`` is the grid import price b[t] and
    ``export_inr_per_kwh[h]`` is the export remuneration a[t] for hour h. The
    same daily schedule applies on every simulated day.
    """

    retail_inr_per_kwh: tuple[float, ...]
    export_inr_per_kwh: tuple[float, ...]

    def __post_init__(self) -> None:
        retail = tuple(
            _non_negative(f"retail price at hour {h}", v) for h, v in enumerate(self.retail_inr_per_kwh)
        )
        export = tuple(
            _non_negative(f"export price at hour {h}", v) for h, v in enumerate(self.export_inr_per_kwh)
        )
        if len(retail) != HOURS_PER_DAY or len(export) != HOURS_PER_DAY:
            raise DataValidationError(
                f"tariff schedule needs exactly {HOURS_PER_DAY} hourly values for retail and export, "
                f"got {len(retail)} and {len(export)}"
            )
        inverted = [h for h in range(HOURS_PER_DAY) if retail[h] < export[h]]
        if inverted:
            warnings.warn(
                f"retail price is below export price at hours {inverted}; no P2P trade can have "
                "positive surplus in those hours",
                EconomicWarning,
                stacklevel=3,
            )
        object.__setattr__(self, "retail_inr_per_kwh", retail)
        object.__setattr__(self, "export_inr_per_kwh", export)

    @classmethod
    def flat(cls, retail_inr_per_kwh: float, export_inr_per_kwh: float) -> TariffSchedule:
        """The same retail and export price in every hour."""
        return cls(
            retail_inr_per_kwh=(retail_inr_per_kwh,) * HOURS_PER_DAY,
            export_inr_per_kwh=(export_inr_per_kwh,) * HOURS_PER_DAY,
        )

    @classmethod
    def from_config(cls, cfg: ScenarioConfig) -> TariffSchedule:
        return cls.flat(cfg.retail_price_inr_per_kwh, cfg.export_price_inr_per_kwh)

    def _check_hour(self, hour: int) -> int:
        if isinstance(hour, bool) or not isinstance(hour, numbers.Integral) or not 0 <= hour < HOURS_PER_DAY:
            raise DataValidationError(f"hour must be an integer in 0..{HOURS_PER_DAY - 1}, got {hour!r}")
        return int(hour)

    def retail_at(self, hour: int) -> float:
        return self.retail_inr_per_kwh[self._check_hour(hour)]

    def export_at(self, hour: int) -> float:
        return self.export_inr_per_kwh[self._check_hour(hour)]


# ---------------------------------------------------------------------------
# Physical state of one hour
# ---------------------------------------------------------------------------


class Role(str, Enum):
    """Market role of a microgrid in one interval (no storage: never both)."""

    SELLER = "seller"
    BUYER = "buyer"
    BALANCED = "balanced"


@dataclass(frozen=True)
class PlayerHour:
    """Energy position of one microgrid in one hourly interval (all kWh)."""

    microgrid_id: str
    generation_kwh: float
    demand_kwh: float
    surplus_kwh: float
    deficit_kwh: float
    role: Role

    def __post_init__(self) -> None:
        object.__setattr__(self, "microgrid_id", _non_empty_str("microgrid_id", self.microgrid_id))
        for name in ("generation_kwh", "demand_kwh", "surplus_kwh", "deficit_kwh"):
            value = getattr(self, name)
            try:
                checked = _non_negative(f"{self.microgrid_id}.{name}", value)
            except DataValidationError as exc:
                raise EnergyBalanceError(str(exc)) from None
            object.__setattr__(self, name, checked)
        if not isinstance(self.role, Role):
            object.__setattr__(self, "role", Role(self.role))

    @property
    def net_kwh(self) -> float:
        """Generation minus demand; positive for a surplus microgrid."""
        return self.generation_kwh - self.demand_kwh


@dataclass(frozen=True)
class HourState:
    """Immutable physical snapshot of every microgrid in one interval.

    Players are stored sorted by ``microgrid_id`` so every downstream
    algorithm sees the same deterministic order.
    """

    day: int
    hour: int
    players: tuple[PlayerHour, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "day", _integer("day", self.day, minimum=0))
        hour = _integer("hour", self.hour, minimum=0)
        if hour >= HOURS_PER_DAY:
            raise DataValidationError(f"hour must be in 0..{HOURS_PER_DAY - 1}, got {hour}")
        object.__setattr__(self, "hour", hour)
        players = tuple(sorted(self.players, key=lambda p: p.microgrid_id))
        if not players:
            raise DataValidationError(f"HourState(day={self.day}, hour={hour}) has no players")
        ids = [p.microgrid_id for p in players]
        if len(set(ids)) != len(ids):
            raise DataValidationError(f"HourState(day={self.day}, hour={hour}) has duplicate ids: {ids}")
        object.__setattr__(self, "players", players)

    @property
    def microgrid_ids(self) -> tuple[str, ...]:
        return tuple(p.microgrid_id for p in self.players)

    def player(self, microgrid_id: str) -> PlayerHour:
        for p in self.players:
            if p.microgrid_id == microgrid_id:
                return p
        raise KeyError(f"microgrid {microgrid_id!r} not present at day {self.day} hour {self.hour}")

    @property
    def sellers(self) -> tuple[str, ...]:
        """IDs with surplus above tolerance, sorted ascending."""
        return tuple(p.microgrid_id for p in self.players if p.role is Role.SELLER)

    @property
    def buyers(self) -> tuple[str, ...]:
        """IDs with deficit above tolerance, sorted ascending."""
        return tuple(p.microgrid_id for p in self.players if p.role is Role.BUYER)

    @property
    def total_generation_kwh(self) -> float:
        return math.fsum(p.generation_kwh for p in self.players)

    @property
    def total_demand_kwh(self) -> float:
        return math.fsum(p.demand_kwh for p in self.players)

    @property
    def total_surplus_kwh(self) -> float:
        return math.fsum(p.surplus_kwh for p in self.players)

    @property
    def total_deficit_kwh(self) -> float:
        return math.fsum(p.deficit_kwh for p in self.players)


# ---------------------------------------------------------------------------
# Market outputs
# ---------------------------------------------------------------------------


TRADE_CSV_COLUMNS = (
    "scenario_id",
    "mechanism",
    "day",
    "hour",
    "seller_id",
    "buyer_id",
    "quantity_kwh",
    "price_inr_per_kwh",
    "seller_fee_inr_per_kwh",
    "buyer_fee_inr_per_kwh",
)

SETTLEMENT_CSV_COLUMNS = (
    "scenario_id",
    "mechanism",
    "day",
    "hour",
    "microgrid_id",
    "grid_import_kwh",
    "grid_export_kwh",
    "p2p_import_kwh",
    "p2p_export_kwh",
    "baseline_cost_inr",
    "p2p_cost_inr",
    "gain_inr",
)

#: Blueprint section 5 solver-diagnostics record, one row per mechanism and interval.
DIAGNOSTICS_CSV_COLUMNS = (
    "scenario_id",
    "mechanism",
    "day",
    "hour",
    "status",
    "pure_ne_found",
    "equilibrium_count",
    "selected_profile",
    "max_unilateral_regret_inr",
    "examined_profiles",
    "not_nash_profile",
    "optimal_welfare_inr",
    "p2p_volume_kwh",
    "message",
)

SETTLEMENT_DETAIL_COLUMNS = (
    "grid_purchase_inr",
    "grid_export_revenue_inr",
    "p2p_purchase_inr",
    "p2p_sales_inr",
    "fees_inr",
)


@dataclass(frozen=True)
class Trade:
    """One bilateral P2P contract in one interval (blueprint section 10).

    Produced by the cooperative and competitive mechanisms (Phase 2+).
    """

    day: int
    hour: int
    seller_id: str
    buyer_id: str
    quantity_kwh: float
    price_inr_per_kwh: float
    seller_fee_inr_per_kwh: float = 0.0
    buyer_fee_inr_per_kwh: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "day", _integer("day", self.day, minimum=0))
        object.__setattr__(self, "hour", _integer("hour", self.hour, minimum=0))
        seller = _non_empty_str("seller_id", self.seller_id)
        buyer = _non_empty_str("buyer_id", self.buyer_id)
        if seller == buyer:
            raise DataValidationError(f"a microgrid cannot trade with itself ({seller})")
        object.__setattr__(self, "seller_id", seller)
        object.__setattr__(self, "buyer_id", buyer)
        for name in (
            "quantity_kwh",
            "price_inr_per_kwh",
            "seller_fee_inr_per_kwh",
            "buyer_fee_inr_per_kwh",
        ):
            object.__setattr__(self, name, _non_negative(name, getattr(self, name)))

    @property
    def payment_inr(self) -> float:
        """Gross payment from buyer to seller: q * p."""
        return self.quantity_kwh * self.price_inr_per_kwh

    @property
    def fee_revenue_inr(self) -> float:
        """Total fees collected by the external operator: q * (f_s + f_b)."""
        return self.quantity_kwh * (self.seller_fee_inr_per_kwh + self.buyer_fee_inr_per_kwh)

    @property
    def seller_net_price_inr_per_kwh(self) -> float:
        """What the seller keeps per kWh: p - f_s."""
        return self.price_inr_per_kwh - self.seller_fee_inr_per_kwh

    @property
    def buyer_delivered_price_inr_per_kwh(self) -> float:
        """What the buyer pays per kWh including its fee: p + f_b."""
        return self.price_inr_per_kwh + self.buyer_fee_inr_per_kwh


@dataclass(frozen=True)
class PlayerSettlement:
    """Signed financial settlement of one microgrid in one interval.

    ``p2p_cost_inr`` keeps the blueprint's CSV column name. It is the
    microgrid's net cost under the reported mechanism, so for the baseline
    it equals ``baseline_cost_inr``.

    Identity: p2p_cost = grid_purchase - grid_export_revenue
                         + p2p_purchase - p2p_sales + fees
    """

    scenario_id: str
    mechanism: str
    day: int
    hour: int
    microgrid_id: str
    grid_import_kwh: float
    grid_export_kwh: float
    p2p_import_kwh: float
    p2p_export_kwh: float
    grid_purchase_inr: float
    grid_export_revenue_inr: float
    p2p_purchase_inr: float
    p2p_sales_inr: float
    fees_inr: float
    baseline_cost_inr: float
    p2p_cost_inr: float
    gain_inr: float

    def __post_init__(self) -> None:
        if self.mechanism not in MECHANISMS:
            raise DataValidationError(f"mechanism must be one of {MECHANISMS}, got {self.mechanism!r}")
        for name in ("grid_import_kwh", "grid_export_kwh", "p2p_import_kwh", "p2p_export_kwh"):
            object.__setattr__(self, name, _non_negative(name, getattr(self, name)))
        for name in (
            "grid_purchase_inr",
            "grid_export_revenue_inr",
            "p2p_purchase_inr",
            "p2p_sales_inr",
            "fees_inr",
            "baseline_cost_inr",
            "p2p_cost_inr",
            "gain_inr",
        ):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))

    def to_record(self, detailed: bool = False) -> dict[str, Any]:
        columns = SETTLEMENT_CSV_COLUMNS + (SETTLEMENT_DETAIL_COLUMNS if detailed else ())
        return {name: getattr(self, name) for name in columns}


@dataclass(frozen=True)
class MarketDiagnostics:
    """Solver/equilibrium status attached to every market result.

    ``status`` is one of:
        "success": P2P trades were settled (for the competitive market, at a
            verified pure Nash equilibrium)
        "no_trade": no voluntary trade exists (baseline-equivalent settlement)
        "NO_PURE_NE": the competitive game has no pure equilibrium
        "ENUMERATION_CAP_EXCEEDED": exact enumeration was not attempted

    ``optimal_welfare_inr`` is the social-LP optimum (cooperative mechanism).

    Competitive mechanism fields:
        ``equilibrium_count``: number of pure equilibria found
        ``selected_profile``: the settled equilibrium as (seller_id, price) pairs
        ``max_regret_inr``: that equilibrium's maximum unilateral regret
        ``not_nash_profile``: for NO_PURE_NE only, the minimum-maximum-regret
            diagnostic profile, which is explicitly NOT a Nash equilibrium;
            ``max_regret_inr`` then holds its (positive) regret
    """

    status: str
    pure_ne_found: bool | None = None
    max_regret_inr: float | None = None
    examined_profiles: int | None = None
    optimal_welfare_inr: float | None = None
    equilibrium_count: int | None = None
    selected_profile: tuple[tuple[str, float], ...] | None = None
    not_nash_profile: tuple[tuple[str, float], ...] | None = None
    message: str = ""


@dataclass(frozen=True)
class MarketResult:
    """Outcome of one mechanism in one interval: trades, settlements, diagnostics."""

    scenario_id: str
    mechanism: str
    day: int
    hour: int
    trades: tuple[Trade, ...]
    settlements: tuple[PlayerSettlement, ...]
    diagnostics: MarketDiagnostics

    def __post_init__(self) -> None:
        if self.mechanism not in MECHANISMS:
            raise DataValidationError(f"mechanism must be one of {MECHANISMS}, got {self.mechanism!r}")
        object.__setattr__(self, "trades", tuple(self.trades))
        settlements = tuple(sorted(self.settlements, key=lambda s: s.microgrid_id))
        for s in settlements:
            if (s.day, s.hour, s.mechanism, s.scenario_id) != (
                self.day,
                self.hour,
                self.mechanism,
                self.scenario_id,
            ):
                raise DataValidationError(
                    f"settlement for {s.microgrid_id} does not belong to this result "
                    f"({s.scenario_id}/{s.mechanism}/day {s.day}/hour {s.hour})"
                )
        ids = [s.microgrid_id for s in settlements]
        if len(set(ids)) != len(ids):
            raise DataValidationError(f"duplicate settlements for {ids}")
        object.__setattr__(self, "settlements", settlements)

    def settlement_for(self, microgrid_id: str) -> PlayerSettlement:
        for s in self.settlements:
            if s.microgrid_id == microgrid_id:
                return s
        raise KeyError(f"no settlement for {microgrid_id!r}")

    @property
    def total_baseline_cost_inr(self) -> float:
        return math.fsum(s.baseline_cost_inr for s in self.settlements)

    @property
    def total_cost_inr(self) -> float:
        return math.fsum(s.p2p_cost_inr for s in self.settlements)

    @property
    def total_gain_inr(self) -> float:
        return math.fsum(s.gain_inr for s in self.settlements)

    @property
    def total_p2p_volume_kwh(self) -> float:
        return math.fsum(t.quantity_kwh for t in self.trades)

    @property
    def total_p2p_payment_inr(self) -> float:
        """Gross money moved between microgrids (sum of q * p)."""
        return math.fsum(t.payment_inr for t in self.trades)

    @property
    def total_fee_revenue_inr(self) -> float:
        """Fees collected by the external operator in this interval."""
        return math.fsum(t.fee_revenue_inr for t in self.trades)

    @property
    def total_grid_import_kwh(self) -> float:
        return math.fsum(s.grid_import_kwh for s in self.settlements)

    @property
    def total_grid_export_kwh(self) -> float:
        return math.fsum(s.grid_export_kwh for s in self.settlements)
