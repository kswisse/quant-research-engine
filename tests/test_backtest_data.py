"""Tests for PriceData."""

from __future__ import annotations

import numpy as np
import pytest

from quant_engine.backtest.data import PriceData
from quant_engine.core.errors import DataError


class TestPriceDataConstruction:
    """Test PriceData creation and validation."""

    def test_valid_construction(self) -> None:
        prices = PriceData(np.array([100.0, 101.0, 102.0]))
        assert prices.n_periods == 3
        np.testing.assert_array_equal(prices.prices, [100.0, 101.0, 102.0])

    def test_timestamps(self) -> None:
        prices = PriceData(np.array([10.0, 20.0, 30.0]))
        np.testing.assert_array_equal(prices.timestamps, [0, 1, 2])

    def test_len(self) -> None:
        prices = PriceData(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert len(prices) == 5

    def test_repr(self) -> None:
        prices = PriceData(np.array([100.0, 101.0, 102.0]))
        r = repr(prices)
        assert "PriceData" in r
        assert "n_periods=3" in r

    def test_minimum_length(self) -> None:
        with pytest.raises(DataError, match="at least 2"):
            PriceData(np.array([100.0]))

    def test_empty_array(self) -> None:
        with pytest.raises(DataError, match="at least 2"):
            PriceData(np.array([]))

    def test_non_1d_rejected(self) -> None:
        with pytest.raises(DataError, match="1-D"):
            PriceData(np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_nan_rejected(self) -> None:
        with pytest.raises(DataError, match="non-finite"):
            PriceData(np.array([100.0, float("nan"), 102.0]))

    def test_inf_rejected(self) -> None:
        with pytest.raises(DataError, match="non-finite"):
            PriceData(np.array([100.0, float("inf"), 102.0]))

    def test_negative_price_rejected(self) -> None:
        with pytest.raises(DataError, match="positive"):
            PriceData(np.array([100.0, -1.0, 102.0]))

    def test_zero_price_rejected(self) -> None:
        with pytest.raises(DataError, match="positive"):
            PriceData(np.array([100.0, 0.0, 102.0]))


class TestReturns:
    """Test simple return computation."""

    def test_simple_returns(self) -> None:
        prices = PriceData(np.array([100.0, 110.0, 121.0]))
        returns = prices.returns()
        np.testing.assert_allclose(returns, [0.10, 0.10])

    def test_returns_length(self) -> None:
        prices = PriceData(np.array([100.0, 105.0, 110.0, 115.0]))
        assert len(prices.returns()) == 3

    def test_declining_returns(self) -> None:
        prices = PriceData(np.array([200.0, 180.0, 162.0]))
        returns = prices.returns()
        np.testing.assert_allclose(returns, [-0.10, -0.10])
