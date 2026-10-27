"""Profile, config-file and tariff-file validation: missing, malformed and invalid records."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microgrid.data import (
    PROFILE_COLUMNS,
    load_and_validate_profiles,
    load_config,
    load_tariffs_csv,
    validate_profiles,
)
from microgrid.models import DataValidationError

HEADER = "day,hour,microgrid_id,generation_kwh,demand_kwh\n"
GOLDEN_ROWS = "0,12,MG1,40,20\n0,12,MG2,10,25\n0,12,MG3,25,15\n0,12,MG4,5,30\n"


def _error(df, cfg, **kwargs) -> DataValidationError:
    with pytest.raises(DataValidationError) as info:
        validate_profiles(df, cfg, **kwargs)
    return info.value


# --- valid input and normalisation ------------------------------------------


def test_valid_rows_pass_with_canonical_types(golden_raw, golden_cfg):
    out = validate_profiles(golden_raw, golden_cfg)
    assert list(out.columns) == list(PROFILE_COLUMNS)
    assert out["microgrid_id"].tolist() == ["MG1", "MG2", "MG3", "MG4"]
    assert str(out["day"].dtype) == "int64" and str(out["hour"].dtype) == "int64"
    assert str(out["generation_kwh"].dtype) == "float64" and str(out["demand_kwh"].dtype) == "float64"


def test_rows_are_sorted_and_ids_are_stripped(golden_raw, golden_cfg):
    shuffled = golden_raw.iloc[[3, 1, 0, 2]].copy()
    shuffled["microgrid_id"] = [" MG4", "MG2 ", "MG1", " MG3 "]
    out = validate_profiles(shuffled, golden_cfg)
    assert out["microgrid_id"].tolist() == ["MG1", "MG2", "MG3", "MG4"]
    assert out.set_index("microgrid_id").loc["MG4", "generation_kwh"] == 5.0


def test_extra_columns_are_dropped(golden_raw, golden_cfg):
    golden_raw["comment"] = "synthetic"
    assert list(validate_profiles(golden_raw, golden_cfg).columns) == list(PROFILE_COLUMNS)


def test_validation_does_not_modify_input(golden_raw, golden_cfg):
    shuffled = golden_raw.iloc[[2, 0, 3, 1]]
    snapshot = shuffled.copy(deep=True)
    validate_profiles(shuffled, golden_cfg)
    pd.testing.assert_frame_equal(shuffled, snapshot)


# --- missing columns / empty input ------------------------------------------


@pytest.mark.parametrize(
    "dropped", [["demand_kwh"], ["day", "generation_kwh"], ["microgrid_id"], list(PROFILE_COLUMNS)]
)
def test_missing_columns_are_reported_exactly(golden_raw, golden_cfg, dropped):
    err = _error(golden_raw.drop(columns=dropped), golden_cfg)
    expected = [c for c in PROFILE_COLUMNS if c in dropped]
    assert f"missing required column(s) {expected}" in str(err)


def test_empty_table_is_rejected(golden_cfg):
    err = _error(pd.DataFrame(columns=list(PROFILE_COLUMNS)), golden_cfg)
    assert "no data rows" in str(err)


def test_non_dataframe_is_rejected(golden_cfg):
    with pytest.raises(TypeError, match="DataFrame"):
        validate_profiles([{"day": 0}], golden_cfg)  # type: ignore[arg-type]


# --- invalid values ---------------------------------------------------------


@pytest.mark.parametrize(
    "column,value,reason",
    [
        ("generation_kwh", np.nan, "missing value"),
        ("demand_kwh", None, "missing value"),
        ("generation_kwh", np.inf, "non-finite"),
        ("demand_kwh", -np.inf, "non-finite"),
        ("generation_kwh", -5.0, "negative energy"),
        ("demand_kwh", -0.001, "negative energy"),
        ("generation_kwh", "abc", "non-numeric"),
    ],
)
def test_invalid_energy_values_name_row_and_column(golden_raw, golden_cfg, column, value, reason):
    bad = golden_raw.astype({column: object})
    bad.loc[1, column] = value
    message = str(_error(bad, golden_cfg))
    assert f"column '{column}'" in message
    assert reason in message
    assert "row 1" in message


@pytest.mark.parametrize(
    "column,value,reason",
    [
        ("hour", 24, "outside 0..23"),
        ("hour", -1, "outside 0..23"),
        ("hour", 3.5, "non-integer"),
        ("hour", np.nan, "missing value"),
        ("hour", "noon", "non-numeric"),
        ("day", -1, "outside 0..0"),
        ("day", 1, "outside 0..0"),  # beyond the configured 1-day horizon
        ("day", 0.5, "non-integer"),
    ],
)
def test_invalid_time_keys_are_rejected(golden_raw, golden_cfg, column, value, reason):
    bad = golden_raw.astype({column: object})
    bad.loc[2, column] = value
    message = str(_error(bad, golden_cfg))
    assert f"column '{column}'" in message
    assert reason in message
    assert "row 2" in message


@pytest.mark.parametrize("value", ["", "   ", None, np.nan])
def test_missing_microgrid_id_is_rejected(golden_raw, golden_cfg, value):
    bad = golden_raw.astype({"microgrid_id": object})
    bad.loc[2, "microgrid_id"] = value
    message = str(_error(bad, golden_cfg))
    assert "column 'microgrid_id': missing or empty id" in message
    assert "row 2" in message


def test_unknown_microgrid_id_is_rejected(golden_raw, golden_cfg):
    golden_raw.loc[0, "microgrid_id"] = "MG9"
    message = str(_error(golden_raw, golden_cfg))
    assert "unknown id(s) ['MG9']" in message


def test_duplicate_records_are_rejected(golden_raw, golden_cfg):
    duplicated = pd.concat([golden_raw, golden_raw.iloc[[0]]], ignore_index=True)
    message = str(_error(duplicated, golden_cfg))
    assert "duplicate record for (day=0, hour=12, microgrid_id='MG1')" in message
    assert "row 0" in message and "row 4" in message


def test_all_problems_are_reported_together(golden_raw, golden_cfg):
    bad = golden_raw.astype(object)
    bad.loc[0, "generation_kwh"] = -1.0
    bad.loc[1, "hour"] = 99
    bad.loc[3, "microgrid_id"] = "ZZ"
    err = _error(bad, golden_cfg)
    assert len(err.problems) == 3
    assert "3 problem(s)" in str(err)


# --- missing records / completeness -----------------------------------------


def test_missing_microgrid_in_an_hour_is_rejected(golden_raw, golden_cfg):
    message = str(_error(golden_raw.drop(index=2), golden_cfg))
    assert "missing 1 record(s)" in message
    assert "(day=0, hour=12, microgrid_id='MG3')" in message


def test_full_horizon_needs_all_96_rows(sample_profiles, default_cfg):
    message = str(_error(sample_profiles.drop(index=50), default_cfg))
    assert "missing 1 record(s)" in message
    assert "needs 96 rows" in message and "found 95" in message


def test_single_hour_fails_when_full_horizon_required(golden_raw, golden_cfg):
    message = str(_error(golden_raw, golden_cfg, require_full_horizon=True))
    assert "missing 92 record(s)" in message  # 23 absent hours x 4 microgrids


def test_partial_horizon_still_needs_every_microgrid(golden_raw, golden_cfg):
    extra_hour = golden_raw.assign(hour=13).iloc[:3]  # hour 13 lacks MG4
    message = str(_error(pd.concat([golden_raw, extra_hour], ignore_index=True), golden_cfg))
    assert "(day=0, hour=13, microgrid_id='MG4')" in message


# --- CSV files --------------------------------------------------------------


def test_csv_errors_report_file_and_line(tmp_path, golden_cfg):
    path = tmp_path / "bad.csv"
    path.write_text(HEADER + GOLDEN_ROWS.replace("0,12,MG2,10,25", "0,12,MG2,-10,25"))
    with pytest.raises(DataValidationError) as info:
        load_and_validate_profiles(path, golden_cfg)
    message = str(info.value)
    assert str(path) in message
    assert "row 1 (CSV line 3) = -10" in message


def test_csv_malformed_number_is_rejected(tmp_path, golden_cfg):
    path = tmp_path / "typo.csv"
    path.write_text(HEADER + GOLDEN_ROWS.replace("0,12,MG1,40,20", "0,12,MG1,4O,20"))
    with pytest.raises(DataValidationError, match="non-numeric value at row 0 \\(CSV line 2\\) = '4O'"):
        load_and_validate_profiles(path, golden_cfg)


def test_csv_missing_column_is_reported(tmp_path, golden_cfg):
    path = tmp_path / "no_demand.csv"
    path.write_text("day,hour,microgrid_id,generation_kwh\n0,12,MG1,40\n")
    with pytest.raises(DataValidationError, match=r"missing required column\(s\) \['demand_kwh'\]"):
        load_and_validate_profiles(path, golden_cfg)


def test_empty_and_header_only_csv_are_rejected(tmp_path, golden_cfg):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(DataValidationError, match="empty"):
        load_and_validate_profiles(empty, golden_cfg)
    header_only = tmp_path / "header.csv"
    header_only.write_text(HEADER)
    with pytest.raises(DataValidationError, match="no data rows"):
        load_and_validate_profiles(header_only, golden_cfg)


def test_missing_profile_file(tmp_path, golden_cfg):
    with pytest.raises(FileNotFoundError):
        load_and_validate_profiles(tmp_path / "nope.csv", golden_cfg)


# --- config and tariff files ------------------------------------------------


def test_invalid_config_files(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    with pytest.raises(DataValidationError, match="invalid JSON"):
        load_config(broken)
    bad_alpha = tmp_path / "alpha.json"
    bad_alpha.write_text('{"bargaining_seller_weight": 0}')
    with pytest.raises(DataValidationError, match="alpha.json"):
        load_config(bad_alpha)
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.json")


def _write_tariffs(path, rows):
    path.write_text(
        "hour,retail_price_inr_per_kwh,export_price_inr_per_kwh\n"
        + "".join(f"{h},{b},{a}\n" for h, b, a in rows)
    )
    return path


def test_tariff_csv_loads_in_hour_order(tmp_path):
    rows = [(h, 8.0 + (2.0 if 17 <= h <= 21 else 0.0), 3.0) for h in reversed(range(24))]
    tariffs = load_tariffs_csv(_write_tariffs(tmp_path / "t.csv", rows))
    assert tariffs.retail_at(18) == 10.0 and tariffs.retail_at(12) == 8.0
    assert tariffs.export_at(0) == 3.0


def test_tariff_csv_errors(tmp_path):
    full = [(h, 8.0, 3.0) for h in range(24)]
    with pytest.raises(DataValidationError, match=r"missing tariff rows for hour\(s\) \[5\]"):
        load_tariffs_csv(_write_tariffs(tmp_path / "a.csv", [r for r in full if r[0] != 5]))
    with pytest.raises(DataValidationError, match="duplicate tariff rows"):
        load_tariffs_csv(_write_tariffs(tmp_path / "b.csv", full + [(3, 8.0, 3.0)]))
    with pytest.raises(DataValidationError, match="finite prices >= 0"):
        load_tariffs_csv(_write_tariffs(tmp_path / "c.csv", full[:-1] + [(23, -8.0, 3.0)]))
    with pytest.raises(DataValidationError, match="integers 0..23"):
        load_tariffs_csv(_write_tariffs(tmp_path / "d.csv", full[:-1] + [(24, 8.0, 3.0)]))
    no_export = tmp_path / "e.csv"
    no_export.write_text("hour,retail_price_inr_per_kwh\n0,8\n")
    with pytest.raises(DataValidationError, match="missing tariff column"):
        load_tariffs_csv(no_export)
