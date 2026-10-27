"""Regression snapshot: seed-42 24-hour results of all three mechanisms (blueprint section 11).

These values were produced by this code (v0.4.0) on ``data/sample_24h.csv``.
They are a change detector, not an independent check; correctness is
covered by the invariant, fixture and reference-enumeration tests. If an
intended model change moves them, re-run

    python -m microgrid.cli compare --config configs/default.json --profiles data/sample_24h.csv

and update the values together with an explanation of the change.
"""

from __future__ import annotations

import pytest

from microgrid.simulation import compare_all

EXPECTED = {
    "baseline": {
        "net_community_cost_inr": 8633.5998,
        "total_gain_inr": 0.0,
        "total_p2p_volume_kwh": 0.0,
        "average_price_inr_per_kwh": None,
        "jain_fairness": None,
        "efficiency": 0.0,
        "trading_intervals": 0,
        "per_player_gain_inr": {"MG1": 0.0, "MG2": 0.0, "MG3": 0.0, "MG4": 0.0},
    },
    "cooperative": {
        "net_community_cost_inr": 7373.6648,
        "total_gain_inr": 1259.935,
        "total_p2p_volume_kwh": 251.987,
        "average_price_inr_per_kwh": 5.5,
        "jain_fairness": 0.8297572412,
        "efficiency": 1.0,
        "trading_intervals": 10,
        "per_player_gain_inr": {"MG1": 520.5445, "MG2": 308.18375, "MG3": 117.19125, "MG4": 314.0155},
    },
    "competitive": {
        "net_community_cost_inr": 7771.5183,
        "total_gain_inr": 862.0815,
        "total_p2p_volume_kwh": 172.4163,
        "average_price_inr_per_kwh": 7.5,
        "jain_fairness": 0.4673344211,
        "efficiency": 0.6842269641,
        "trading_intervals": 8,
        "per_player_gain_inr": {"MG1": 602.8954, "MG2": 46.70605, "MG3": 174.5316, "MG4": 37.94845},
    },
}


@pytest.fixture(scope="module")
def comparison():
    from microgrid.data import load_and_validate_profiles, load_config
    from conftest import REPO_ROOT

    cfg = load_config(REPO_ROOT / "configs" / "default.json")
    return compare_all(load_and_validate_profiles(REPO_ROOT / "data" / "sample_24h.csv", cfg), cfg)


@pytest.mark.parametrize("mechanism", sorted(EXPECTED))
def test_seed42_24h_snapshot(comparison, mechanism):
    metrics = comparison.metrics[mechanism]
    for key, expected in EXPECTED[mechanism].items():
        if expected is None:
            assert metrics[key] is None, key
        elif isinstance(expected, dict):
            assert metrics[key] == pytest.approx(expected, abs=1e-6), key
        else:
            assert metrics[key] == pytest.approx(expected, abs=1e-6), key
