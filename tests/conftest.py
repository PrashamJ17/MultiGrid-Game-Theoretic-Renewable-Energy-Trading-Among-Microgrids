"""Shared pytest fixtures: repository paths, configs and validated profile tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from microgrid.data import load_and_validate_profiles, load_config
from microgrid.energy import compute_hour_states
from microgrid.models import HourState, ScenarioConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def golden_cfg() -> ScenarioConfig:
    return load_config(REPO_ROOT / "configs" / "golden_one_hour.json")


@pytest.fixture
def default_cfg() -> ScenarioConfig:
    return load_config(REPO_ROOT / "configs" / "default.json")


@pytest.fixture
def golden_profiles(golden_cfg: ScenarioConfig) -> pd.DataFrame:
    return load_and_validate_profiles(REPO_ROOT / "data" / "golden_one_hour.csv", golden_cfg)


@pytest.fixture
def golden_state(golden_profiles: pd.DataFrame, golden_cfg: ScenarioConfig) -> HourState:
    states = compute_hour_states(golden_profiles, golden_cfg.tol)
    assert len(states) == 1
    return states[0]


@pytest.fixture
def sample_profiles(default_cfg: ScenarioConfig) -> pd.DataFrame:
    return load_and_validate_profiles(REPO_ROOT / "data" / "sample_24h.csv", default_cfg)


@pytest.fixture
def golden_raw() -> pd.DataFrame:
    """Unvalidated golden rows, for tests that corrupt them."""
    return pd.DataFrame(
        {
            "day": [0, 0, 0, 0],
            "hour": [12, 12, 12, 12],
            "microgrid_id": ["MG1", "MG2", "MG3", "MG4"],
            "generation_kwh": [40.0, 10.0, 25.0, 5.0],
            "demand_kwh": [20.0, 25.0, 15.0, 30.0],
        }
    )


@pytest.fixture(scope="session")
def golden_expected() -> dict[str, dict[str, float]]:
    """Blueprint section 4 golden values (kWh and INR)."""
    return {
        "MG1": {"generation": 40.0, "demand": 20.0, "surplus": 20.0, "deficit": 0.0, "baseline_cost": -60.0},
        "MG2": {"generation": 10.0, "demand": 25.0, "surplus": 0.0, "deficit": 15.0, "baseline_cost": 120.0},
        "MG3": {"generation": 25.0, "demand": 15.0, "surplus": 10.0, "deficit": 0.0, "baseline_cost": -30.0},
        "MG4": {"generation": 5.0, "demand": 30.0, "surplus": 0.0, "deficit": 25.0, "baseline_cost": 200.0},
    }
