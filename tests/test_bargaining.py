"""Bilateral weighted Nash bargaining: closed form, its optimality, and trade eligibility (T03)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from microgrid.cooperative import bargaining_price, is_beneficial_trade, nash_product, unit_trade_surplus


def test_golden_bargaining_price():  # blueprint reference test
    assert bargaining_price(3.0, 8.0, 0.0, 0.0, 0.5) == pytest.approx(5.5)


def test_fee_fixture_price():
    price = bargaining_price(3.0, 8.0, 1.0, 1.0, 0.5)
    assert price == pytest.approx(5.5)
    assert price - 1.0 == pytest.approx(4.5)  # seller net receipt
    assert price + 1.0 == pytest.approx(6.5)  # buyer delivered price


@pytest.mark.parametrize("alpha,expected", [(0.1, 3.5), (0.25, 4.25), (0.5, 5.5), (0.75, 6.75)])
def test_alpha_moves_the_price_towards_the_buyer_reservation(alpha, expected):
    assert bargaining_price(3.0, 8.0, 0.0, 0.0, alpha) == pytest.approx(expected)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_alpha_must_be_strictly_inside_the_unit_interval(alpha):
    with pytest.raises(ValueError, match="alpha"):
        bargaining_price(3.0, 8.0, 0.0, 0.0, alpha)


@pytest.mark.parametrize("f_s,f_b", [(2.5, 2.5), (3.0, 2.0), (5.0, 0.0), (4.0, 4.0)])
def test_trade_without_bargaining_surplus_is_rejected(f_s, f_b):
    with pytest.raises(ValueError, match="no positive bargaining surplus"):
        bargaining_price(3.0, 8.0, f_s, f_b, 0.5)


def test_positive_trade_surplus_is_required():  # blueprint reference test
    assert not is_beneficial_trade(8.0, 3.0, 3.0, 2.0)
    assert not is_beneficial_trade(8.0, 3.0, 2.5, 2.5)
    assert is_beneficial_trade(8.0, 3.0, 2.4, 2.5)
    assert is_beneficial_trade(8.0, 3.0)
    assert not is_beneficial_trade(3.0, 3.0)  # no spread
    assert not is_beneficial_trade(2.0, 3.0)  # inverted tariffs
    assert unit_trade_surplus(8.0, 3.0, 1.0, 1.0) == pytest.approx(3.0)


@pytest.mark.parametrize(
    "a,b,f_s,f_b,alpha",
    [(3.0, 8.0, 0.0, 0.0, 0.5), (3.0, 8.0, 1.0, 1.0, 0.5), (3.0, 8.0, 0.0, 0.0, 0.3), (2.5, 9.0, 0.4, 0.2, 0.8), (3.0, 8.0, 0.5, 1.5, 0.6)],
)
def test_closed_form_maximises_the_weighted_nash_product(a, b, f_s, f_b, alpha):
    q = 7.0
    r_s, r_b = a + f_s, b - f_b
    closed = bargaining_price(a, b, f_s, f_b, alpha)
    numeric = minimize_scalar(
        lambda p: -math.log(nash_product(p, q, a, b, f_s, f_b, alpha)),
        bounds=(r_s + 1e-9, r_b - 1e-9),
        method="bounded",
        options={"xatol": 1e-10},
    )
    assert numeric.x == pytest.approx(closed, abs=1e-6)
    best_on_grid = max(nash_product(p, q, a, b, f_s, f_b, alpha) for p in np.linspace(r_s, r_b, 2001))
    assert best_on_grid <= nash_product(closed, q, a, b, f_s, f_b, alpha) + 1e-9


@pytest.mark.parametrize("alpha", [0.2, 0.5, 0.9])
def test_seller_receives_share_alpha_of_the_trade_surplus(alpha):
    price = bargaining_price(3.0, 8.0, 1.0, 1.0, alpha)
    seller_gain = price - (3.0 + 1.0)
    buyer_gain = (8.0 - 1.0) - price
    assert seller_gain > 0 and buyer_gain > 0
    assert seller_gain / (seller_gain + buyer_gain) == pytest.approx(alpha)


def test_nash_product_is_zero_outside_the_bargaining_set():
    assert nash_product(3.0, 10.0, 3.0, 8.0) == 0.0  # seller gains nothing
    assert nash_product(8.0, 10.0, 3.0, 8.0) == 0.0  # buyer gains nothing
    assert nash_product(9.0, 10.0, 3.0, 8.0) == 0.0
    assert nash_product(5.5, 10.0, 3.0, 8.0) == pytest.approx(25.0)  # sqrt(25 * 25)
