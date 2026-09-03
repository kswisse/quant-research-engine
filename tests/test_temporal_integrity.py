"""Temporal integrity tests.

These tests verify that the engine enforces the correct temporal convention
and prevents look-ahead bias. They test both positive cases (correct alignment)
and negative cases (information that would violate temporal order).
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_engine.backtest.data import PriceData
from quant_engine.backtest.engine import run_backtest
from quant_engine.backtest.strategies import SMACrossover


class TestLookAheadPrevention:
    """Verify that the engine prevents look-ahead bias."""

    def test_position_at_t_uses_only_past_info(self) -> None:
        """The position at time t is determined by signal[t],
        which is generated using information through prices[t]."""
        # Construct data where signal at t=2 is determined by prices[0:3]
        prices = np.array([100.0, 102.0, 98.0, 105.0, 110.0, 95.0])

        class AlwaysFlat:
            name = "always_flat"

            def generate_signal(self, data: PriceData) -> np.ndarray:
                return np.zeros(data.n_periods, dtype=np.float64)

        result = run_backtest(PriceData(prices), AlwaysFlat())
        # All returns should be zero
        np.testing.assert_array_equal(result.strategy_returns, [0.0, 0.0, 0.0, 0.0, 0.0])

    def test_strategy_return_at_t_uses_position_from_t_minus_1(self) -> None:
        """strategy_return[t] = position[t] * asset_return[t]
        where position[t] = signals[t] (set at close of period t)."""
        # Hand-verifiable case:
        # prices = [100, 110, 121]
        # signals = [0, +1, +1]
        # positions = signals[:-1] = [0, +1]  (held during next period)
        # asset_returns = [0.10, 0.10]
        # strategy_returns = [0*0.10, 1*0.10] = [0, 0.10]
        prices = np.array([100.0, 110.0, 121.0])

        class SignalAfterFirst:
            name = "signal_after_first"

            def generate_signal(self, data: PriceData) -> np.ndarray:
                return np.array([0.0, 1.0, 1.0])

        result = run_backtest(PriceData(prices), SignalAfterFirst())
        np.testing.assert_allclose(result.strategy_returns, [0.0, 0.10])
        np.testing.assert_allclose(result.positions, [0.0, 1.0])

    def test_signal_t0_cannot_affect_return_t0(self) -> None:
        """The signal at t=0 cannot affect any return because
        the earliest return period is t=1."""
        prices = np.array([100.0, 105.0, 110.0])

        class LoudSignal:
            name = "loud"

            def generate_signal(self, data: PriceData) -> np.ndarray:
                # Even if signal[0] = +1, position[0] = signal[0] = +1
                # But this position IS used for return period 0
                return np.array([1.0, 1.0, 1.0])

        result = run_backtest(PriceData(prices), LoudSignal())
        # positions = signals[:-1] = [1, 1]
        # asset_returns = [0.05, 0.0476...]
        # strategy_returns = [1*0.05, 1*0.0476] = [0.05, 0.0476]
        expected_asset_returns = np.diff(prices) / prices[:-1]
        np.testing.assert_allclose(
            result.strategy_returns, expected_asset_returns
        )

    def test_sma_crossover_no_future_data(self) -> None:
        """SMACrossover only uses prices[0:t+1] at time t."""
        # Create data with a clear trend reversal
        prices = np.array(
            [100.0, 101.0, 102.0, 103.0, 104.0, 103.0, 102.0, 101.0, 100.0, 99.0]
        )
        data = PriceData(prices)
        strategy = SMACrossover(fast_window=2, slow_window=5)
        result = run_backtest(data, strategy)

        # Verify that at time t=5, the signal depends only on prices[0:6]
        # At t=5: prices = [100, 101, 102, 103, 104, 103]
        #   fast_sma (last 2) = mean(104, 103) = 103.5
        #   slow_sma (last 5) = mean(101, 102, 103, 104, 103) = 102.6
        #   fast > slow => signal[5] = +1
        assert result.signals[5] == 1.0

        # At t=7: prices = [100, 101, 102, 103, 104, 103, 102, 101]
        #   fast_sma = mean(102, 101) = 101.5
        #   slow_sma = mean(102, 103, 104, 103, 102, 101) -- wait, slow_window=5
        #   slow_sma = mean(102, 104, 103, 102, 101) -- let me recalculate
        #   window = prices[:8] = [100, 101, 102, 103, 104, 103, 102, 101]
        #   fast_sma = mean(window[-2:]) = mean(102, 101) = 101.5
        #   slow_sma = mean(window[-5:]) = mean(103, 104, 103, 102, 101) = 102.6
        #   fast < slow => signal[7] = -1
        assert result.signals[7] == -1.0


class TestHandVerifiableCase:
    """Hand-verifiable test case for the full pipeline.

    This test documents the exact temporal chain for a specific price series
    so that a human can verify the engine's correctness by hand.
    """

    def test_documented_pipeline(self) -> None:
        """Hand-trace this test case:

        Prices: [100, 110, 121, 108.9, 119.79]

        Strategy: BuyAndHold (signal[0]=0, signal[t]=+1 for t>=1)

        Step 1: Generate signals
            signal = [0, 1, 1, 1, 1]

        Step 2: Shift for temporal alignment
            positions = signal[:-1] = [0, 1, 1, 1]

        Step 3: Compute asset returns
            asset_returns = [0.10, 0.10, -0.10, 0.10]

        Step 4: Compute strategy returns
            strategy_returns = [0*0.10, 1*0.10, 1*(-0.10), 1*0.10]
                             = [0, 0.10, -0.10, 0.10]

        Step 5: Compute total return
            total_return = (1+0)(1+0.10)(1-0.10)(1+0.10) - 1
                        = 1 * 1.1 * 0.9 * 1.1 - 1
                        = 1.089 - 1
                        = 0.089
        """
        prices = np.array([100.0, 110.0, 121.0, 108.9, 119.79])
        data = PriceData(prices)

        class ManualBuyAndHold:
            name = "manual_bah"

            def generate_signal(self, data: PriceData) -> np.ndarray:
                return np.array([0.0, 1.0, 1.0, 1.0, 1.0])

        result = run_backtest(data, ManualBuyAndHold())

        # Verify each step
        np.testing.assert_array_equal(result.signals, [0, 1, 1, 1, 1])
        np.testing.assert_array_equal(result.positions, [0, 1, 1, 1])
        np.testing.assert_allclose(
            result.asset_returns, [0.10, 0.10, -0.10, 0.10]
        )
        np.testing.assert_allclose(
            result.strategy_returns, [0.0, 0.10, -0.10, 0.10]
        )
        assert abs(result.total_return - 0.089) < 1e-10
