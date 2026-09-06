"""Order book domain models for prediction market research.

Public API:
    OrderBookLevel: Immutable price level (price, size)
    OrderBookSnapshot: Immutable depth-aware order book observation
    snapshot_id: Deterministic identity for snapshots
    validate_snapshot: Validate an order book snapshot
    best_bid: Best bid price
    best_ask: Best ask price
    spread: Top-of-book spread
    cumulative_depth: Cumulative depth for first N levels
    DepthLevel: Cumulative depth result type
"""

from quant_engine.order_book.calculations import (
    DepthLevel,
    best_ask,
    best_bid,
    cumulative_depth,
    spread,
)
from quant_engine.order_book.models import (
    OrderBookLevel,
    OrderBookSnapshot,
    snapshot_id,
)
from quant_engine.order_book.validation import validate_snapshot

__all__ = [
    "DepthLevel",
    "OrderBookLevel",
    "OrderBookSnapshot",
    "best_ask",
    "best_bid",
    "cumulative_depth",
    "snapshot_id",
    "spread",
    "validate_snapshot",
]
