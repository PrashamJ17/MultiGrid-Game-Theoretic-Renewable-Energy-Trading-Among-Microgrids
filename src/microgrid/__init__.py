"""MultiGrid: game-theoretic simulation of P2P renewable energy trading among microgrids.

Implemented so far:
    Phase 1: data contracts, synthetic profiles, energy model, utility-only baseline.
    Phase 2: shared settlement and validation, plus the cooperative market
    (welfare-maximising LP allocation with bilateral weighted Nash bargaining).
    Phase 3: the competitive market, a finite seller price game with
    exhaustive pure-Nash verification (``NO_PURE_NE`` when none exists).
    Phase 4: ``simulation.compare_all`` (every mechanism on the same frozen
    states), ``metrics.compute_metrics`` (explicit N/A handling), the
    diagnostics export and the Streamlit dashboard (``app.py``).

All data and tariffs are synthetic or illustrative. This is a research
simulation, not a trading or grid-control system.
"""

__version__ = "0.4.0"

from .baseline import baseline_cost, community_baseline_cost, run_baseline, solve_baseline
from .competitive import analyze_price_game, run_competitive, solve_competitive
from .cooperative import bargaining_price, is_beneficial_trade, run_cooperative, solve_cooperative
from .data import generate_profiles, load_and_validate_profiles, load_config, validate_profiles
from .energy import compute_hour_states, deficit, surplus
from .metrics import compute_metrics, jain_fairness
from .models import (
    CombinatorialWarning,
    DataValidationError,
    EconomicWarning,
    EnergyBalanceError,
    HourState,
    MarketDiagnostics,
    MarketResult,
    PlayerHour,
    PlayerSettlement,
    Role,
    ScenarioConfig,
    SettlementError,
    SolverError,
    TariffSchedule,
    Trade,
)
from .settlement import settle_market, validate_market
from .simulation import ComparisonResult, compare_all

__all__ = [
    "__version__",
    "CombinatorialWarning",
    "ComparisonResult",
    "DataValidationError",
    "EconomicWarning",
    "EnergyBalanceError",
    "HourState",
    "MarketDiagnostics",
    "MarketResult",
    "PlayerHour",
    "PlayerSettlement",
    "Role",
    "ScenarioConfig",
    "SettlementError",
    "SolverError",
    "TariffSchedule",
    "Trade",
    "analyze_price_game",
    "bargaining_price",
    "baseline_cost",
    "community_baseline_cost",
    "compare_all",
    "compute_metrics",
    "compute_hour_states",
    "deficit",
    "generate_profiles",
    "is_beneficial_trade",
    "jain_fairness",
    "load_and_validate_profiles",
    "load_config",
    "run_baseline",
    "run_competitive",
    "run_cooperative",
    "settle_market",
    "solve_baseline",
    "solve_competitive",
    "solve_cooperative",
    "surplus",
    "validate_market",
    "validate_profiles",
]
