"""Jain fairness index: golden value and zero-denominator / negative-gain handling."""

from __future__ import annotations

import pytest

from microgrid.metrics import jain_fairness


def test_golden_fairness():
    assert jain_fairness([50.0, 37.5, 25.0, 37.5]) == pytest.approx(0.94736842, abs=1e-8)


def test_equal_gains_are_perfectly_fair():
    assert jain_fairness([10.0, 10.0, 10.0]) == pytest.approx(1.0)


def test_one_winner_gives_one_over_n():
    assert jain_fairness([12.0, 0.0, 0.0, 0.0]) == pytest.approx(0.25)


def test_no_gains_is_not_applicable():
    assert jain_fairness([0.0, 0.0, 0.0]) is None
    assert jain_fairness([]) is None
    assert jain_fairness([1e-12, 0.0]) is None


def test_negative_gains_are_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        jain_fairness([10.0, -1.0])
    assert jain_fairness([10.0, -1e-12]) == pytest.approx(0.5)  # float drift within tolerance is clipped
    with pytest.raises(ValueError, match="finite"):
        jain_fairness([float("nan"), 1.0])
