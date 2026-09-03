"""Property-based tests using Hypothesis.

These tests verify invariants that must hold for ANY valid strategy and data.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_engine.backtest.data import PriceData
from quant_engine.backtest.engine import run_backtest


@st.composite
def price_data_st(draw: st.DrawFn) -> PriceData:
    """Generate valid PriceData with random prices."""
    n = draw(st.integers(min_value=2, max_value=200))
    # Generate random positive prices
    prices = draw(
        st.lists(
            st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    return PriceData(np.array(prices))


@st.composite
def strategy_signals_st(draw: st.DrawFn) -> np.ndarray:
    """Generate valid signal arrays."""
    n = draw(st.integers(min_value=2, max_value=200))
    signals = draw(
        st.lists(
            st.sampled_from([-1.0, 0.0, 1.0]),
            min_size=n,
            max_size=n,
        )
    )
    return np.array(signals)


class DummyStrategy:
    """Strategy that accepts externally generated signals."""

    def __init__(self, signals: np.ndarray) -> None:
        self.name = "dummy"
        self._signals = signals

    def generate_signal(self, data: PriceData) -> np.ndarray:
        return self._signals


@given(data=price_data_st(), signals=strategy_signals_st())
@settings(max_examples=200, deadline=1000)
def test_strategy_returns_finite(data: PriceData, signals: np.ndarray) -> None:
    """Strategy returns must always be finite."""
    # Ensure signals match price length
    signals = signals[: data.n_periods]
    signals = np.pad(signals, (0, max(0, data.n_periods - len(signals))), constant_values=0.0)

    result = run_backtest(data, DummyStrategy(signals))
    assert np.all(np.isfinite(result.strategy_returns))


@given(data=price_data_st(), signals=strategy_signals_st())
@settings(max_examples=200, deadline=1000)
def test_output_array_lengths_consistent(data: PriceData, signals: np.ndarray) -> None:
    """All output arrays must have consistent lengths."""
    signals = signals[: data.n_periods]
    signals = np.pad(signals, (0, max(0, data.n_periods - len(signals))), constant_values=0.0)

    result = run_backtest(data, DummyStrategy(signals))
    n_return = data.n_periods - 1
    assert len(result.asset_returns) == n_return
    assert len(result.strategy_returns) == n_return
    assert len(result.positions) == n_return
    assert len(result.timestamps) == n_return


@given(data=price_data_st(), signals=strategy_signals_st())
@settings(max_examples=200, deadline=1000)
def test_strategy_return_bounded_by_asset_return(
    data: PriceData, signals: np.ndarray
) -> None:
    """|strategy_return[t]| <= |asset_return[t]| for all t
    when signals are in {-1, 0, +1}."""
    signals = signals[: data.n_periods]
    signals = np.pad(signals, (0, max(0, data.n_periods - len(signals))), constant_values=0.0)

    result = run_backtest(data, DummyStrategy(signals))
    # positions are in {-1, 0, 1}, so |strategy_return| <= |asset_return|
    assert np.all(np.abs(result.strategy_returns) <= np.abs(result.asset_returns) + 1e-10)


@given(data=price_data_st(), signals=strategy_signals_st())
@settings(max_examples=200, deadline=1000)
def test_positions_are_shifted_signals(data: PriceData, signals: np.ndarray) -> None:
    """positions[t] == signals[t] for the return-period mapping."""
    signals = signals[: data.n_periods]
    signals = np.pad(signals, (0, max(0, data.n_periods - len(signals))), constant_values=0.0)

    result = run_backtest(data, DummyStrategy(signals))
    np.testing.assert_array_equal(result.positions, signals[:-1])


@given(data=price_data_st())
@settings(max_examples=200, deadline=1000)
def test_zero_signal_always_zero_return(data: PriceData) -> None:
    """A strategy that always returns flat should always have zero returns."""
    zero_signals = np.zeros(data.n_periods, dtype=np.float64)
    result = run_backtest(data, DummyStrategy(zero_signals))
    np.testing.assert_array_equal(result.strategy_returns, [0.0] * (data.n_periods - 1))
