"""Tests for Strategy protocol and synthetic strategies."""

from __future__ import annotations

import numpy as np
import pytest

from quant_engine.backtest.data import PriceData
from quant_engine.backtest.strategies import (
    AlwaysFlat,
    AlwaysShort,
    BuyAndHold,
    SMACrossover,
)
from quant_engine.backtest.strategy import Strategy


class TestStrategyProtocol:
    """Test that strategies conform to the Protocol."""

    def test_buy_and_hold_is_strategy(self) -> None:
        assert isinstance(BuyAndHold(), Strategy)

    def test_always_flat_is_strategy(self) -> None:
        assert isinstance(AlwaysFlat(), Strategy)

    def test_always_short_is_strategy(self) -> None:
        assert isinstance(AlwaysShort(), Strategy)

    def test_sma_crossover_is_strategy(self) -> None:
        assert isinstance(SMACrossover(), Strategy)


class TestBuyAndHold:
    """Test BuyAndHold strategy."""

    def test_signal_length(self) -> None:
        data = PriceData(np.array([100.0, 101.0, 102.0, 103.0, 104.0]))
        strategy = BuyAndHold()
        signals = strategy.generate_signal(data)
        assert len(signals) == 5

    def test_first_signal_is_flat(self) -> None:
        data = PriceData(np.array([100.0, 101.0, 102.0]))
        strategy = BuyAndHold()
        signals = strategy.generate_signal(data)
        assert signals[0] == 0.0

    def test_subsequent_signals_are_long(self) -> None:
        data = PriceData(np.array([100.0, 101.0, 102.0, 103.0]))
        strategy = BuyAndHold()
        signals = strategy.generate_signal(data)
        np.testing.assert_array_equal(signals[1:], [1.0, 1.0, 1.0])


class TestAlwaysFlat:
    """Test AlwaysFlat strategy."""

    def test_all_zeros(self) -> None:
        data = PriceData(np.array([100.0, 101.0, 102.0, 103.0]))
        strategy = AlwaysFlat()
        signals = strategy.generate_signal(data)
        np.testing.assert_array_equal(signals, [0.0, 0.0, 0.0, 0.0])


class TestAlwaysShort:
    """Test AlwaysShort strategy."""

    def test_first_signal_is_flat(self) -> None:
        data = PriceData(np.array([100.0, 101.0, 102.0]))
        strategy = AlwaysShort()
        signals = strategy.generate_signal(data)
        assert signals[0] == 0.0

    def test_subsequent_signals_are_short(self) -> None:
        data = PriceData(np.array([100.0, 101.0, 102.0, 103.0]))
        strategy = AlwaysShort()
        signals = strategy.generate_signal(data)
        np.testing.assert_array_equal(signals[1:], [-1.0, -1.0, -1.0])


class TestSMACrossover:
    """Test SMACrossover strategy."""

    def test_invalid_fast_window(self) -> None:
        with pytest.raises(ValueError, match="fast_window"):
            SMACrossover(fast_window=0)

    def test_invalid_slow_window(self) -> None:
        with pytest.raises(ValueError, match="slow_window"):
            SMACrossover(slow_window=0)

    def test_fast_gte_slow_rejected(self) -> None:
        with pytest.raises(ValueError, match="fast_window.*must be < slow"):
            SMACrossover(fast_window=20, slow_window=5)

    def test_no_signals_before_slow_window(self) -> None:
        data = PriceData(np.arange(1, 21, dtype=np.float64))  # 1..20
        strategy = SMACrossover(fast_window=3, slow_window=5)
        signals = strategy.generate_signal(data)
        # First slow_window signals should be 0
        np.testing.assert_array_equal(signals[:5], [0.0, 0.0, 0.0, 0.0, 0.0])

    def test_trending_up_gives_long_signal(self) -> None:
        # Monotonically increasing prices
        prices = np.arange(1, 30, dtype=np.float64)
        data = PriceData(prices)
        strategy = SMACrossover(fast_window=3, slow_window=10)
        signals = strategy.generate_signal(data)
        # After slow_window, all signals should be +1 (fast > slow)
        np.testing.assert_array_equal(signals[10:], np.ones(19))

    def test_trending_down_gives_short_signal(self) -> None:
        # Monotonically decreasing prices
        prices = np.arange(30, 1, -1, dtype=np.float64)
        data = PriceData(prices)
        strategy = SMACrossover(fast_window=3, slow_window=10)
        signals = strategy.generate_signal(data)
        # After slow_window, all signals should be -1 (fast < slow)
        np.testing.assert_array_equal(signals[10:], -np.ones(19))
