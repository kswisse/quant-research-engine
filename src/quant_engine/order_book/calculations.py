"""Pure order book calculations.

Provides deterministic, provider-independent calculations
for order book analysis. No execution semantics.

All functions are pure: same input → same output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from quant_engine.order_book.models import OrderBookSnapshot


@dataclass(frozen=True)
class DepthLevel:
    """A price level with cumulative depth information.

    Attributes:
        price: Price at this level.
        size: Size at this level.
        cumulative_size: Sum of sizes from best price through this level.
        cumulative_notional: Sum of price * size through this level.
    """

    price: float
    size: float
    cumulative_size: float
    cumulative_notional: float


def best_bid(snapshot: OrderBookSnapshot) -> float | None:
    """Best bid price (highest). None if no bids.

    Args:
        snapshot: The order book snapshot.

    Returns:
        Highest bid price, or None.
    """
    return snapshot.bids[0].price if snapshot.bids else None


def best_ask(snapshot: OrderBookSnapshot) -> float | None:
    """Best ask price (lowest). None if no asks.

    Args:
        snapshot: The order book snapshot.

    Returns:
        Lowest ask price, or None.
    """
    return snapshot.asks[0].price if snapshot.asks else None


def spread(snapshot: OrderBookSnapshot) -> float | None:
    """Top-of-book spread (best_ask - best_bid). None if either side is empty.

    Args:
        snapshot: The order book snapshot.

    Returns:
        Spread as a float, or None.
    """
    b = best_bid(snapshot)
    a = best_ask(snapshot)
    if b is not None and a is not None:
        return a - b
    return None


def cumulative_depth(
    snapshot: OrderBookSnapshot,
    side: Literal["bid", "ask"],
    levels: int,
) -> list[DepthLevel]:
    """Cumulative depth for the first N levels on a side.

    For bids: starts from best bid (highest price) downward.
    For asks: starts from best ask (lowest price) upward.

    Args:
        snapshot: The order book snapshot.
        side: "bid" or "ask".
        levels: Number of levels to include. Must be > 0.

    Returns:
        List of DepthLevel with cumulative information.
        Returns empty list if the side has no levels.

    Raises:
        ValueError: If levels <= 0.
    """
    if levels <= 0:
        raise ValueError(f"levels must be > 0, got {levels}")

    book_side = snapshot.bids if side == "bid" else snapshot.asks
    if not book_side:
        return []

    result: list[DepthLevel] = []
    cumulative_size = 0.0
    cumulative_notional = 0.0

    for level in book_side[:levels]:
        cumulative_size += level.size
        cumulative_notional += level.price * level.size
        result.append(
            DepthLevel(
                price=level.price,
                size=level.size,
                cumulative_size=cumulative_size,
                cumulative_notional=cumulative_notional,
            )
        )

    return result
