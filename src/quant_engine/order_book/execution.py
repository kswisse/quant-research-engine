"""Mechanical order book execution simulator.

Provides deterministic, provider-independent simulation of consuming
displayed liquidity from a static order book.

This is a mechanical calculation — it does NOT model:
- future liquidity
- queue position
- fill probability
- latency
- adverse selection
- market impact beyond displayed depth
- fees
- slippage assumptions
- price movement during execution

Architecture:
    OrderBookSnapshot → consume_book() → ExecutionResult
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from quant_engine.order_book.models import OrderBookSnapshot


@dataclass(frozen=True)
class ExecutionFill:
    """A single fill at a specific price level.

    Attributes:
        price: Price at which the fill occurred.
        size: Number of contracts filled at this level.
    """

    price: float
    size: float

    @property
    def notional(self) -> float:
        """Total notional value at this fill (price * size)."""
        return self.price * self.size


@dataclass(frozen=True)
class ExecutionResult:
    """Result of mechanical order book consumption.

    Attributes:
        side: "buy" or "sell".
        requested_size: Original requested number of contracts.
        filled_size: Number of contracts actually filled.
        remaining_size: Number of contracts not filled (requested - filled).
        total_notional: Sum of fill notionals (price * size).
        vwap: Volume-weighted average price. None if nothing fills.
        fills: Per-level fill details, in execution order.
        fully_filled: True if requested_size was completely filled.
    """

    side: Literal["buy", "sell"]
    requested_size: float
    filled_size: float
    remaining_size: float
    total_notional: float
    vwap: float | None
    fills: tuple[ExecutionFill, ...]
    fully_filled: bool


def consume_book(
    snapshot: OrderBookSnapshot,
    side: Literal["buy", "sell"],
    requested_size: float,
) -> ExecutionResult:
    """Mechanical execution: consume displayed liquidity.

    For a BUY order: consumes asks from lowest to highest price.
    For a SELL order: consumes bids from highest to lowest price.

    This is a pure, deterministic calculation. It consumes displayed
    liquidity mechanically and does NOT validate the snapshot's health.
    Callers who need health guarantees should call validate_snapshot()
    separately.

    Args:
        snapshot: The order book snapshot to consume.
        side: "buy" to consume asks, "sell" to consume bids.
        requested_size: Number of contracts requested. Must be > 0 and finite.

    Returns:
        ExecutionResult with fill details.

    Raises:
        ValueError: If requested_size is <= 0, NaN, or infinity.
        ValueError: If side is not "buy" or "sell".
    """
    # Validate inputs
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    if requested_size <= 0 or math.isnan(requested_size) or math.isinf(requested_size):
        raise ValueError(f"requested_size must be > 0 and finite, got {requested_size}")

    # Select the book side to consume
    book_side = snapshot.asks if side == "buy" else snapshot.bids

    # Consume levels
    remaining = requested_size
    fills: list[ExecutionFill] = []
    total_notional = 0.0

    for level in book_side:
        if remaining <= 0:
            break

        filled = min(remaining, level.size)
        fill = ExecutionFill(price=level.price, size=filled)
        fills.append(fill)
        total_notional += fill.notional
        remaining -= filled

    filled_size = requested_size - remaining
    vwap = total_notional / filled_size if filled_size > 0 else None

    return ExecutionResult(
        side=side,
        requested_size=requested_size,
        filled_size=filled_size,
        remaining_size=remaining,
        total_notional=total_notional,
        vwap=vwap,
        fills=tuple(fills),
        fully_filled=remaining == 0,
    )
