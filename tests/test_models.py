"""Config, tariff and data-model validation (input validation for non-profile inputs)."""

from __future__ import annotations

import dataclasses
import warnings

import pytest

from microgrid.models import (
    DataValidationError,
    EconomicWarning,
    GeneratorConfig,
    HourState,
    MarketDiagnostics,
    MarketResult,
    MicrogridProfileSpec,
    PlayerHour,
    PlayerSettlement,
    Role,
    ScenarioConfig,
    TariffSchedule,
    Trade,
)

BLUEPRINT_DEFAULT = {
    "scenario_id": "four_microgrids_default",
    "seed": 42,
    "days": 1,
    "hours_per_day": 24,
    "microgrid_ids": ["MG1", "MG2", "MG3", "MG4"],
    "retail_price_inr_per_kwh": 8.0,
    "export_price_inr_per_kwh": 3.0,
    "seller_fee_inr_per_kwh": 0.0,
    "buyer_fee_inr_per_kwh": 0.0,
    "price_step_inr": 0.5,
    "bargaining_seller_weight": 0.5,
    "tol": 1e-8,
    "max_competitive_sellers": 3,
}


# --- ScenarioConfig ---------------------------------------------------------


def test_default_config_file_matches_blueprint(default_cfg):
    data = default_cfg.to_dict()
    for key, value in BLUEPRINT_DEFAULT.items():
        assert data[key] == value, key
    assert default_cfg.expected_rows == 96


def test_config_round_trip(default_cfg, golden_cfg):
    assert ScenarioConfig.from_dict(default_cfg.to_dict()) == default_cfg
    assert ScenarioConfig.from_dict(golden_cfg.to_dict()) == golden_cfg


def test_config_is_immutable(default_cfg):
    with pytest.raises(dataclasses.FrozenInstanceError):
        default_cfg.seed = 7  # type: ignore[misc]


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("bargaining_seller_weight", 0.0, "strictly between 0 and 1"),
        ("bargaining_seller_weight", 1.0, "strictly between 0 and 1"),
        ("bargaining_seller_weight", 1.5, "strictly between 0 and 1"),
        ("retail_price_inr_per_kwh", -1.0, ">= 0"),
        ("export_price_inr_per_kwh", float("nan"), "finite"),
        ("buyer_fee_inr_per_kwh", float("inf"), "finite"),
        ("seller_fee_inr_per_kwh", -0.5, ">= 0"),
        ("retail_price_inr_per_kwh", "8", "real number"),
        ("tol", 0.0, "tol must satisfy"),
        ("tol", -1e-8, "tol must satisfy"),
        ("price_step_inr", 0.0, "price_step_inr must be > 0"),
        ("hours_per_day", 12, "hours_per_day must be 24"),
        ("days", 0, "days must be >= 1"),
        ("seed", -1, "seed must be >= 0"),
        ("seed", 1.5, "seed must be an integer"),
        ("seed", True, "seed must be an integer"),
        ("max_competitive_sellers", 0, "max_competitive_sellers"),
        ("require_full_horizon", "yes", "require_full_horizon"),
        ("microgrid_ids", ["MG1", "MG1"], "duplicates"),
        ("microgrid_ids", ["MG1", "   "], "non-empty"),
        ("microgrid_ids", [], "at least one"),
        ("microgrid_ids", "MG1", "list of strings"),
        ("scenario_id", "", "scenario_id"),
    ],
)
def test_invalid_config_values_are_rejected(field, value, match):
    data = ScenarioConfig().to_dict()
    data[field] = value
    with pytest.raises(DataValidationError, match=match):
        ScenarioConfig.from_dict(data)


def test_unknown_config_key_is_rejected():
    data = ScenarioConfig().to_dict()
    data["retail_price"] = 8.0  # typo of retail_price_inr_per_kwh
    with pytest.raises(DataValidationError, match="unknown config key"):
        ScenarioConfig.from_dict(data)


def test_retail_below_export_warns():
    with pytest.warns(EconomicWarning, match="retail price is below export price"):
        ScenarioConfig(retail_price_inr_per_kwh=2.0, export_price_inr_per_kwh=3.0)


def test_valid_config_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ScenarioConfig()


def test_generator_override_for_unknown_microgrid_is_rejected():
    data = ScenarioConfig().to_dict()
    data["generator"] = {
        "microgrids": {"MG9": {"demand_template": "campus", "solar_peak_kwh": 1.0, "demand_peak_kwh": 1.0}}
    }
    with pytest.raises(DataValidationError, match="unknown microgrid ids"):
        ScenarioConfig.from_dict(data)


def test_generator_rejects_bad_parameters():
    with pytest.raises(DataValidationError, match="demand_template"):
        MicrogridProfileSpec("hospital", 1.0, 1.0)
    with pytest.raises(DataValidationError, match=">= 0"):
        MicrogridProfileSpec("campus", -1.0, 1.0)
    with pytest.raises(DataValidationError, match="sunrise_hour < sunset_hour"):
        GeneratorConfig(sunrise_hour=18.0, sunset_hour=6.0)
    with pytest.raises(DataValidationError, match="noise_std_fraction"):
        GeneratorConfig(noise_std_fraction=0.9)
    with pytest.raises(DataValidationError, match="unknown generator key"):
        GeneratorConfig.from_dict({"solar_scal": 1.0})


# --- TariffSchedule ---------------------------------------------------------


def test_flat_tariff_schedule():
    tariffs = TariffSchedule.flat(8.0, 3.0)
    assert len(tariffs.retail_inr_per_kwh) == 24 and len(tariffs.export_inr_per_kwh) == 24
    assert tariffs.retail_at(0) == 8.0 and tariffs.export_at(23) == 3.0


def test_tariff_from_config(golden_cfg):
    tariffs = TariffSchedule.from_config(golden_cfg)
    assert tariffs.retail_at(12) == 8.0 and tariffs.export_at(12) == 3.0


@pytest.mark.parametrize("hour", [24, -1, 3.0, True])
def test_tariff_rejects_invalid_hour(hour):
    with pytest.raises(DataValidationError, match="hour"):
        TariffSchedule.flat(8.0, 3.0).retail_at(hour)


def test_tariff_requires_24_values_and_non_negative_prices():
    with pytest.raises(DataValidationError, match="exactly 24"):
        TariffSchedule((8.0,) * 23, (3.0,) * 23)
    with pytest.raises(DataValidationError, match=">= 0"):
        TariffSchedule((8.0,) * 23 + (-1.0,), (3.0,) * 24)


def test_inverted_tariff_hour_warns():
    retail = (8.0,) * 24
    export = (3.0,) * 5 + (9.0,) + (3.0,) * 18
    with pytest.warns(EconomicWarning, match=r"hours \[5\]"):
        TariffSchedule(retail, export)


# --- Trade, HourState, results ----------------------------------------------


def test_trade_validation_and_money():
    trade = Trade(0, 12, "MG1", "MG2", 15.0, 5.5, 1.0, 1.0)
    assert trade.payment_inr == pytest.approx(82.5)
    assert trade.fee_revenue_inr == pytest.approx(30.0)
    with pytest.raises(DataValidationError, match="itself"):
        Trade(0, 12, "MG1", "MG1", 1.0, 5.0)
    with pytest.raises(DataValidationError, match="quantity_kwh"):
        Trade(0, 12, "MG1", "MG2", -1.0, 5.0)


def _player(mg: str, g: float, d: float, role: Role) -> PlayerHour:
    return PlayerHour(mg, g, d, max(g - d, 0.0), max(d - g, 0.0), role)


def test_hour_state_sorts_players_and_lists_roles():
    state = HourState(
        day=0,
        hour=12,
        players=(
            _player("MG4", 5, 30, Role.BUYER),
            _player("MG1", 40, 20, Role.SELLER),
            _player("MG3", 25, 15, Role.SELLER),
            _player("MG2", 10, 25, Role.BUYER),
        ),
    )
    assert state.microgrid_ids == ("MG1", "MG2", "MG3", "MG4")
    assert state.sellers == ("MG1", "MG3")
    assert state.buyers == ("MG2", "MG4")
    with pytest.raises(KeyError, match="MG9"):
        state.player("MG9")


def test_hour_state_rejects_duplicates_and_bad_hours():
    p = _player("MG1", 1, 1, Role.BALANCED)
    with pytest.raises(DataValidationError, match="duplicate"):
        HourState(0, 0, (p, p))
    with pytest.raises(DataValidationError, match="hour"):
        HourState(0, 24, (p,))


def test_player_hour_rejects_negative_energy():
    from microgrid.models import EnergyBalanceError

    with pytest.raises(EnergyBalanceError, match="generation_kwh"):
        PlayerHour("MG1", -1.0, 0.0, 0.0, 1.0, Role.BUYER)


def _settlement(**overrides) -> PlayerSettlement:
    values = dict(
        scenario_id="s",
        mechanism="baseline",
        day=0,
        hour=0,
        microgrid_id="MG1",
        grid_import_kwh=0.0,
        grid_export_kwh=0.0,
        p2p_import_kwh=0.0,
        p2p_export_kwh=0.0,
        grid_purchase_inr=0.0,
        grid_export_revenue_inr=0.0,
        p2p_purchase_inr=0.0,
        p2p_sales_inr=0.0,
        fees_inr=0.0,
        baseline_cost_inr=0.0,
        p2p_cost_inr=0.0,
        gain_inr=0.0,
    )
    values.update(overrides)
    return PlayerSettlement(**values)


def test_market_result_rejects_unknown_mechanism_and_foreign_settlements():
    diag = MarketDiagnostics(status="success")
    with pytest.raises(DataValidationError, match="mechanism"):
        MarketResult("s", "auction", 0, 0, (), (), diag)
    with pytest.raises(DataValidationError, match="does not belong"):
        MarketResult("s", "baseline", 0, 1, (), (_settlement(),), diag)
    with pytest.raises(DataValidationError, match="grid_import_kwh"):
        _settlement(grid_import_kwh=-1.0)
