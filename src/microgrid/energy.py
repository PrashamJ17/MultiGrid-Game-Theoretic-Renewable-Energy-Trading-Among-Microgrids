"""Physical energy model: surplus/deficit classification and conservation checks.

Responsibilities
    * S = max(G - D, 0) and R = max(D - G, 0) for scalars, arrays or Series.
    * Classify each microgrid as seller, buyer or balanced. There is no
      storage, so a microgrid is never both.
    * Build immutable ``HourState`` snapshots from a validated profile table.
    * Check the per-player energy identity
          G + B + P2P_in = D + X + P2P_out
      with non-negative flows, never importing and exporting in the same
      interval, and never buying and selling P2P in the same interval.

Interaction
    Consumes validated frames from ``data``. Produces ``HourState`` objects
    (defined in ``models``), which every market mechanism takes as input.
    ``baseline`` (and later ``settlement``) call ``check_player_balance``
    before returning a result.
"""

from __future__ import annotations

import math
from typing import TypeVar

import numpy as np
import pandas as pd

from .models import EnergyBalanceError, HourState, PlayerHour, Role

DEFAULT_TOL = 1e-8
_REQUIRED = ("day", "hour", "microgrid_id", "generation_kwh", "demand_kwh")

EnergyLike = TypeVar("EnergyLike", float, np.ndarray, pd.Series)


def surplus(generation: EnergyLike, demand: EnergyLike) -> EnergyLike:
    """S = max(G - D, 0) in kWh. Works on scalars, arrays and Series."""
    result = np.maximum(np.subtract(generation, demand), 0.0)
    if isinstance(result, pd.Series) or np.ndim(result) > 0:
        return result
    return float(result)


def deficit(generation: EnergyLike, demand: EnergyLike) -> EnergyLike:
    """R = max(D - G, 0) in kWh. Works on scalars, arrays and Series."""
    result = np.maximum(np.subtract(demand, generation), 0.0)
    if isinstance(result, pd.Series) or np.ndim(result) > 0:
        return result
    return float(result)


def classify(surplus_kwh: float, deficit_kwh: float, tol: float = DEFAULT_TOL) -> Role:
    """Seller if S > tol, buyer if R > tol, otherwise balanced."""
    if surplus_kwh > tol and deficit_kwh > tol:
        raise EnergyBalanceError("a microgrid cannot have surplus and deficit in the same interval")
    if surplus_kwh > tol:
        return Role.SELLER
    if deficit_kwh > tol:
        return Role.BUYER
    return Role.BALANCED


def clamp_tol(value: float, tol: float = DEFAULT_TOL, name: str = "value") -> float:
    """Remove floating-point drift around zero.

    Values in [-tol, 0) become 0.0. Values below -tol are real errors and
    raise ``EnergyBalanceError``. Non-finite values also raise.
    """
    value = float(value)
    if not math.isfinite(value):
        raise EnergyBalanceError(f"{name} is not finite ({value})")
    if value >= 0.0:
        return value
    if value >= -tol:
        return 0.0
    raise EnergyBalanceError(f"{name} = {value} is negative beyond tolerance {tol}")


def add_surplus_deficit_columns(profiles: pd.DataFrame, tol: float = DEFAULT_TOL) -> pd.DataFrame:
    """Return a copy with ``net_kwh``, ``surplus_kwh``, ``deficit_kwh`` and ``role`` columns."""
    out = profiles.copy()
    g = out["generation_kwh"].astype("float64")
    d = out["demand_kwh"].astype("float64")
    out["net_kwh"] = g - d
    out["surplus_kwh"] = surplus(g, d)
    out["deficit_kwh"] = deficit(g, d)
    out["role"] = np.select(
        [out["surplus_kwh"] > tol, out["deficit_kwh"] > tol],
        [Role.SELLER.value, Role.BUYER.value],
        default=Role.BALANCED.value,
    )
    return out


def compute_hour_state(rows: pd.DataFrame, tol: float = DEFAULT_TOL) -> HourState:
    """Build the immutable physical snapshot for one (day, hour) interval."""
    missing = [c for c in _REQUIRED if c not in rows.columns]
    if missing:
        raise ValueError(f"compute_hour_state: missing column(s) {missing}")
    if rows.empty:
        raise ValueError("compute_hour_state: no rows given")
    days = rows["day"].unique()
    hours = rows["hour"].unique()
    if len(days) != 1 or len(hours) != 1:
        raise ValueError(
            f"compute_hour_state: rows span more than one interval "
            f"(days {sorted(days.tolist())}, hours {sorted(hours.tolist())})"
        )
    players = []
    for rec in rows.sort_values("microgrid_id").itertuples(index=False):
        g = float(rec.generation_kwh)
        d = float(rec.demand_kwh)
        if not (math.isfinite(g) and math.isfinite(d)) or g < 0 or d < 0:
            raise EnergyBalanceError(
                f"{rec.microgrid_id}: generation/demand must be finite and >= 0 (got G={g}, D={d})"
            )
        s = surplus(g, d)
        r = deficit(g, d)
        players.append(
            PlayerHour(
                microgrid_id=str(rec.microgrid_id),
                generation_kwh=g,
                demand_kwh=d,
                surplus_kwh=s,
                deficit_kwh=r,
                role=classify(s, r, tol),
            )
        )
    return HourState(day=int(days[0]), hour=int(hours[0]), players=tuple(players))


def compute_hour_states(profiles: pd.DataFrame, tol: float = DEFAULT_TOL) -> list[HourState]:
    """One ``HourState`` per (day, hour), in chronological order."""
    return [
        compute_hour_state(group, tol)
        for _, group in profiles.groupby(["day", "hour"], sort=True)
    ]


def check_player_balance(
    *,
    microgrid_id: str,
    generation_kwh: float,
    demand_kwh: float,
    grid_import_kwh: float,
    grid_export_kwh: float,
    p2p_import_kwh: float = 0.0,
    p2p_export_kwh: float = 0.0,
    tol: float = DEFAULT_TOL,
) -> None:
    """Raise ``EnergyBalanceError`` unless one player's interval is physically consistent.

    Checks:
        * every flow is finite and >= -tol
        * |(G + B + P2P_in) - (D + X + P2P_out)| <= tol * max(1, sum of |flows|)
        * not (B > tol and X > tol): no simultaneous grid import and export
        * not (P2P_in > tol and P2P_out > tol): no circular P2P trading
    """
    flows = {
        "generation_kwh": generation_kwh,
        "demand_kwh": demand_kwh,
        "grid_import_kwh": grid_import_kwh,
        "grid_export_kwh": grid_export_kwh,
        "p2p_import_kwh": p2p_import_kwh,
        "p2p_export_kwh": p2p_export_kwh,
    }
    for name, value in flows.items():
        if not math.isfinite(value):
            raise EnergyBalanceError(f"{microgrid_id}: {name} is not finite ({value})")
        if value < -tol:
            raise EnergyBalanceError(f"{microgrid_id}: {name} = {value} kWh is negative")

    supply = generation_kwh + grid_import_kwh + p2p_import_kwh
    use = demand_kwh + grid_export_kwh + p2p_export_kwh
    residual = supply - use
    scale = max(1.0, sum(abs(v) for v in flows.values()))
    if abs(residual) > tol * scale:
        raise EnergyBalanceError(
            f"{microgrid_id}: energy balance violated, G + B + P2P_in - (D + X + P2P_out) "
            f"= {residual:.3e} kWh (tolerance {tol * scale:.1e})"
        )
    if grid_import_kwh > tol and grid_export_kwh > tol:
        raise EnergyBalanceError(
            f"{microgrid_id}: imports ({grid_import_kwh}) and exports ({grid_export_kwh}) "
            "grid energy in the same interval"
        )
    if p2p_import_kwh > tol and p2p_export_kwh > tol:
        raise EnergyBalanceError(
            f"{microgrid_id}: buys ({p2p_import_kwh}) and sells ({p2p_export_kwh}) P2P energy "
            "in the same interval"
        )
