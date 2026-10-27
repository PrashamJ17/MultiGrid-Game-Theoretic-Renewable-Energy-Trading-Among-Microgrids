"""Seeded synthetic-data generator: reproducibility, shape and physical plausibility (T14)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from microgrid.data import (
    DEMAND_TEMPLATES,
    KEY_COLUMNS,
    PROFILE_COLUMNS,
    generate_profiles,
    load_and_validate_profiles,
    save_profiles,
    solar_shape,
)
from microgrid.energy import compute_hour_states
from microgrid.models import ScenarioConfig

NIGHT_HOURS = [0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23]


def test_same_seed_gives_identical_profiles(default_cfg):
    pd.testing.assert_frame_equal(generate_profiles(default_cfg), generate_profiles(default_cfg))


def test_different_seed_gives_different_values(default_cfg):
    a = generate_profiles(default_cfg)
    b = generate_profiles(dataclasses.replace(default_cfg, seed=43))
    pd.testing.assert_frame_equal(a[list(KEY_COLUMNS)], b[list(KEY_COLUMNS)])
    assert not a.equals(b)


def test_generator_does_not_touch_global_random_state(default_cfg):
    np.random.seed(123)
    expected = np.random.random(5)
    np.random.seed(123)
    generate_profiles(default_cfg)
    np.testing.assert_array_equal(np.random.random(5), expected)


def test_shape_keys_and_order(default_cfg):
    profiles = generate_profiles(default_cfg)
    assert list(profiles.columns) == list(PROFILE_COLUMNS)
    assert len(profiles) == default_cfg.expected_rows == 96
    assert not profiles.duplicated(list(KEY_COLUMNS)).any()
    pd.testing.assert_frame_equal(profiles, profiles.sort_values(list(KEY_COLUMNS)).reset_index(drop=True))
    assert profiles.groupby("hour").size().tolist() == [4] * 24


def test_night_generation_is_exactly_zero_and_demand_positive(default_cfg):
    profiles = generate_profiles(default_cfg)
    assert (profiles.loc[profiles.hour.isin(NIGHT_HOURS), "generation_kwh"] == 0.0).all()
    assert (profiles.loc[profiles.hour.between(7, 16), "generation_kwh"] > 0.0).all()
    assert (profiles.demand_kwh > 0.0).all()


def test_scenario_has_sellers_and_buyers_in_the_same_hour(default_cfg):
    states = compute_hour_states(generate_profiles(default_cfg), default_cfg.tol)
    mixed_hours = [s.hour for s in states if s.sellers and s.buyers]
    assert 12 in mixed_hours
    noon = states[12]
    assert noon.sellers == ("MG1", "MG3") and noon.buyers == ("MG2", "MG4")


def test_committed_sample_matches_generator(default_cfg, repo_root):
    committed = load_and_validate_profiles(repo_root / "data" / "sample_24h.csv", default_cfg)
    regenerated = generate_profiles(default_cfg)
    try:
        pd.testing.assert_frame_equal(committed, regenerated, check_exact=False, rtol=0, atol=1e-9)
    except AssertionError as exc:
        pytest.fail(
            "data/sample_24h.csv no longer matches generate_profiles(seed=42) under NumPy "
            f"{np.__version__}. NumPy does not guarantee identical random streams across versions; "
            "regenerate the file with `python -m microgrid.cli generate --config configs/default.json "
            f"--out data/sample_24h.csv` and record the NumPy version.\n{exc}"
        )


def test_saved_csv_is_byte_identical_and_round_trips(default_cfg, tmp_path):
    profiles = generate_profiles(default_cfg)
    first = save_profiles(profiles, tmp_path / "a.csv")
    second = save_profiles(generate_profiles(default_cfg), tmp_path / "b.csv")
    assert first.read_bytes() == second.read_bytes()
    pd.testing.assert_frame_equal(load_and_validate_profiles(first, default_cfg), profiles)


def test_solar_scale_keeps_profile_shape(default_cfg):
    base = generate_profiles(default_cfg)
    scaled_cfg = dataclasses.replace(
        default_cfg, generator=dataclasses.replace(default_cfg.generator, solar_scale=2.0)
    )
    doubled = generate_profiles(scaled_cfg)
    np.testing.assert_allclose(doubled.generation_kwh, 2.0 * base.generation_kwh, rtol=0, atol=2e-4)
    pd.testing.assert_series_equal(doubled.demand_kwh, base.demand_kwh)
    no_sun_cfg = dataclasses.replace(
        default_cfg, generator=dataclasses.replace(default_cfg.generator, solar_scale=0.0)
    )
    assert (generate_profiles(no_sun_cfg).generation_kwh == 0.0).all()


def test_six_microgrids_and_multiple_days():
    six = ScenarioConfig(microgrid_ids=tuple(f"MG{i}" for i in range(1, 7)))
    assert len(generate_profiles(six)) == 144
    two_days = ScenarioConfig(days=2)
    profiles = generate_profiles(two_days)
    assert len(profiles) == 192
    day0 = profiles[profiles.day == 0].reset_index(drop=True)
    day1 = profiles[profiles.day == 1].reset_index(drop=True)
    assert not np.allclose(day0.demand_kwh, day1.demand_kwh)  # independent noise per day


def test_demand_templates_are_normalised():
    assert set(DEMAND_TEMPLATES) == {"residential", "campus", "commercial", "industrial"}
    for name, template in DEMAND_TEMPLATES.items():
        assert len(template) == 24, name
        assert max(template) == 1.0, name
        assert min(template) > 0.0, name


def test_solar_shape_is_smooth_daylight_curve():
    shape = solar_shape()
    assert shape.shape == (24,)
    assert shape.max() == pytest.approx(1.0)
    assert (shape[NIGHT_HOURS] == 0.0).all()
    np.testing.assert_allclose(shape, shape[::-1], atol=1e-12)  # symmetric about solar noon
    assert (np.diff(shape[6:12]) > 0).all()  # rises through the morning
