"""Strategy abstraction.

Design decision: Strategy is a Protocol requiring a single method:
`generate_signal(data: PriceData) -> np.ndarray`

This keeps strategies decoupled from the engine. The engine calls
the strategy, receives signals, and handles temporal alignment.

Signals are in {-1, 0, +1} (short, flat, long).
Fractional sizing is not supported yet.

Why Protocol instead of ABC:
- Protocols are structural (duck-typing compatible)
- Strategies don't need to inherit from a base class
- Easier to test with mock strategies
- Supports existing classes without modification
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

    from quant_engine.backtest.data import PriceData


@runtime_checkable
class Strategy(Protocol):
    """Protocol for trading strategies.

    A strategy receives price data and returns a signal array.
    The signal array has the same length as the price array.

    Signal convention:
        signal[t] is generated using information available through price[t].
        The engine uses signal[t] to determine position[t].
        The return for period t+1 is: position[t] * asset_return[t+1].

    Signal values:
        +1 = long
         0 = flat
        -1 = short

    The strategy must NOT use future information.
    """

    name: str

    def generate_signal(self, data: PriceData) -> np.ndarray:
        """Generate trading signals from price data.

        Args:
            data: Historical price data.

        Returns:
            1-D array of signals in {-1, 0, +1}, same length as data.prices.
            signal[i] uses information through data.prices[i] only.
        """
        ...
