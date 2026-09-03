"""Minimal market data abstraction.

Design decision: Simple timestamp + close price representation.

The initial engine operates on close prices only. OHLCV can be added later
if needed for more sophisticated strategies. Close-only keeps the architecture
clean and the temporal convention unambiguous.

Timestamps are integer indices (0, 1, 2, ...) representing sequential periods.
This avoids timezone complexity while preserving temporal ordering.
"""

from __future__ import annotations

import numpy as np

from quant_engine.core.errors import DataError


class PriceData:
    """Immutable sequence of prices with integer timestamps.

    Timestamps are implicit: index i corresponds to period i.
    Prices[i] is the closing price at the end of period i.

    Temporal convention:
        - Prices[i] is known at the END of period i.
        - A signal generated at period i uses information through Prices[i].
        - The return for period i+1 is Prices[i+1]/Prices[i] - 1.
    """

    def __init__(self, prices: np.ndarray) -> None:
        """Create PriceData from a 1-D array of prices.

        Args:
            prices: 1-D array of closing prices. Must have >= 2 observations.

        Raises:
            DataError: If prices are invalid.
        """
        arr = np.asarray(prices, dtype=np.float64)
        if arr.ndim != 1:
            raise DataError(f"Expected 1-D array, got {arr.ndim}-D")
        if len(arr) < 2:
            raise DataError(f"Need at least 2 prices, got {len(arr)}")
        if not np.all(np.isfinite(arr)):
            raise DataError("Prices contain non-finite values (NaN, inf)")
        if np.any(arr <= 0):
            raise DataError("Prices must be positive")
        # Check for duplicate consecutive prices (not strictly an error, but flag)
        self._prices = arr

    @property
    def prices(self) -> np.ndarray:
        """The price array (read-only access)."""
        return self._prices

    @property
    def n_periods(self) -> int:
        """Number of price observations."""
        return len(self._prices)

    @property
    def timestamps(self) -> np.ndarray:
        """Integer timestamp array [0, 1, ..., n_periods-1]."""
        return np.arange(self.n_periods)

    def returns(self) -> np.ndarray:
        """Compute simple returns: r[t] = prices[t+1]/prices[t] - 1.

        Returns array of length n_periods - 1.
        Returns[t] is the return earned from holding through period t to t+1.
        """
        return np.diff(self._prices) / self._prices[:-1]

    def __len__(self) -> int:
        return self.n_periods

    def __repr__(self) -> str:
        first = self._prices[0]
        last = self._prices[-1]
        return (
            f"PriceData(n_periods={self.n_periods}, "
            f"first={first:.4f}, last={last:.4f})"
        )
