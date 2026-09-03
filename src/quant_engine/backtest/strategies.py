"""Deterministic synthetic strategies for testing.

These are intentionally simple strategies used for:
- verifying engine correctness
- hand-verifiable test cases
- baseline comparisons

None of these strategies are profitable by construction.
They exist to test the engine's temporal alignment and correctness.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from quant_engine.backtest.data import PriceData


class BuyAndHold:
    """Always long. After the first observation, signal is always +1.

    signal[0] = 0 (no position on first observation)
    signal[t] = +1 for t >= 1
    """

    name: str = "buy_and_hold"

    def generate_signal(self, data: PriceData) -> np.ndarray:
        n = data.n_periods
        signals = np.ones(n, dtype=np.float64)
        signals[0] = 0.0  # flat on first observation (no prior data)
        return signals


class AlwaysFlat:
    """Always flat. No exposure ever.

    signal[t] = 0 for all t.
    """

    name: str = "always_flat"

    def generate_signal(self, data: PriceData) -> np.ndarray:
        return np.zeros(data.n_periods, dtype=np.float64)


class AlwaysShort:
    """Always short. After the first observation, signal is always -1.

    signal[0] = 0 (no position on first observation)
    signal[t] = -1 for t >= 1
    """

    name: str = "always_short"

    def generate_signal(self, data: PriceData) -> np.ndarray:
        n = data.n_periods
        signals = -np.ones(n, dtype=np.float64)
        signals[0] = 0.0
        return signals


class SMACrossover:
    """Simple Moving Average crossover.

    signal[t] = +1 if fast_sma[t] > slow_sma[t]
    signal[t] = -1 if fast_sma[t] < slow_sma[t]
    signal[t] =  0 if fast_sma[t] == slow_sma[t]

    Uses only information available through prices[t].
    No look-ahead: SMAs are computed on prices[0:t+1] only.
    """

    name: str = "sma_crossover"

    def __init__(self, fast_window: int = 5, slow_window: int = 20) -> None:
        if fast_window < 1:
            raise ValueError(f"fast_window must be >= 1, got {fast_window}")
        if slow_window < 1:
            raise ValueError(f"slow_window must be >= 1, got {slow_window}")
        if fast_window >= slow_window:
            raise ValueError(
                f"fast_window ({fast_window}) must be < slow_window ({slow_window})"
            )
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.name = f"sma_crossover_{fast_window}_{slow_window}"

    def generate_signal(self, data: PriceData) -> np.ndarray:
        prices = data.prices
        n = len(prices)
        signals = np.zeros(n, dtype=np.float64)

        for t in range(self.slow_window, n):
            # Compute SMAs using only prices[0:t+1]
            window = prices[: t + 1]
            fast_sma = np.mean(window[-self.fast_window :])
            slow_sma = np.mean(window[-self.slow_window :])
            if fast_sma > slow_sma:
                signals[t] = 1.0
            elif fast_sma < slow_sma:
                signals[t] = -1.0

        return signals
