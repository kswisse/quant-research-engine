"""Tests for the backtest engine."""

from __future__ import annotations

import numpy as np
import pytest

from quant_engine.backtest.data import PriceData
from quant_engine.backtest.engine import BacktestOutput, run_backtest
from quant_engine.backtest.strategies import (
    AlwaysFlat,
    AlwaysShort,
    BuyAndHold,
    SMACrossover,
)


class TestEngineOutput:
    """Test that the engine produces correct outputs."""

    def test_output_shape(self) -> None:
        data = PriceData(np.array([100.0, 101.0, 102.0, 103.0, 104.0]))
        result = run_backtest(data, BuyAndHold())
        assert len(result.asset_returns) == 4
        assert len(result.strategy_returns) == 4
        assert len(result.positions) == 4
        assert len(result.signals) == 5  # signals has full length
        assert len(result.timestamps) == 4

    def test_asset_returns_computation(self) -> None:
        data = PriceData(np.array([100.0, 110.0, 121.0]))
        result = run_backtest(data, BuyAndHold())
        np.testing.assert_allclose(result.asset_returns, [0.10, 0.10])

    def test_timestamps_are_consecutive(self) -> None:
        data = PriceData(np.array([100.0, 101.0, 102.0, 103.0]))
        result = run_backtest(data, BuyAndHold())
        np.testing.assert_array_equal(result.timestamps, [1, 2, 3])

    def test_strategy_name(self) -> None:
        data = PriceData(np.array([100.0, 101.0, 102.0]))
        result = run_backtest(data, BuyAndHold())
        assert result.strategy_name == "buy_and_hold"

    def test_zero_cost(self) -> None:
        data = PriceData(np.array([100.0, 101.0, 102.0]))
        result = run_backtest(data, BuyAndHold())
        assert result.cost_assumption == "zero_cost"
        np.testing.assert_array_equal(result.transaction_costs, [0.0, 0.0])


class TestTemporalAlignment:
    """Verify temporal alignment is correct."""

    def test_buy_and_hold_returns_match_asset_returns(self) -> None:
        data = PriceData(np.array([100.0, 110.0, 121.0, 133.1]))
        result = run_backtest(data, BuyAndHold())
        # BuyAndHold: signal=[0, 1, 1, 1], position=[0, 1, 1]
        # strategy_returns = position * asset_returns = [0*0.10, 1*0.10, 1*0.10]
        np.testing.assert_allclose(
            result.strategy_returns, [0.0, 0.10, 0.10]
        )

    def test_always_flat_returns_zero(self) -> None:
        data = PriceData(np.array([100.0, 110.0, 121.0]))
        result = run_backtest(data, AlwaysFlat())
        np.testing.assert_allclose(result.strategy_returns, [0.0, 0.0])

    def test_always_short_inverses_returns(self) -> None:
        data = PriceData(np.array([100.0, 110.0, 121.0, 108.9]))
        result = run_backtest(data, AlwaysShort())
        # AlwaysShort: signal=[0, -1, -1, -1], position=[0, -1, -1]
        # asset_returns = [0.10, 0.10, -0.10]
        # strategy_returns = [0*-0.10, -1*0.10, -1*-0.10] = [0, -0.10, 0.10]
        np.testing.assert_allclose(
            result.strategy_returns, [0.0, -0.10, 0.10]
        )

    def test_total_return(self) -> None:
        data = PriceData(np.array([100.0, 110.0, 121.0]))
        result = run_backtest(data, BuyAndHold())
        # First return period: position=0, so no return
        # Second return period: position=1, return=0.10
        # total_return = (1+0)*(1+0.10) - 1 = 0.10
        assert abs(result.total_return - 0.10) < 1e-10


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_rejects_wrong_signal_length(self) -> None:
        class BadStrategy:
            name = "bad"

            def generate_signal(self, data: PriceData) -> np.ndarray:
                return np.zeros(data.n_periods + 1)  # wrong length

        data = PriceData(np.array([100.0, 101.0, 102.0]))
        with pytest.raises(ValueError, match="Signal length"):
            run_backtest(data, BadStrategy())

    def test_rejects_invalid_signal_values(self) -> None:
        class BadStrategy:
            name = "bad"

            def generate_signal(self, data: PriceData) -> np.ndarray:
                return np.full(data.n_periods, 0.5)

        data = PriceData(np.array([100.0, 101.0, 102.0]))
        with pytest.raises(ValueError, match="Signals must be in"):
            run_backtest(data, BadStrategy())

    def test_minimum_prices(self) -> None:
        data = PriceData(np.array([100.0, 101.0]))
        result = run_backtest(data, BuyAndHold())
        assert len(result.strategy_returns) == 1
