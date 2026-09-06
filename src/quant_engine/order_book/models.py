"""Order book domain models for prediction market research.

Provider-independent representation of order book depth.
All timestamps are UTC, timezone-aware. Price in [0.00, 1.00] range.
Size represents contracts (not USD notional).

Architecture:
    MarketQuote  = canonical top-of-book observation
    OrderBookSnapshot = depth-aware research representation

The order book model does NOT contain provider-specific logic.
Provider adapters are responsible for converting raw data to these models.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime  # noqa: TC003 — used at runtime for isoformat()
from typing import Any

from pydantic import BaseModel, Field, model_validator


class OrderBookLevel(BaseModel):
    """Immutable representation of one price level in an order book.

    Attributes:
        price: Price in [0.00, 1.00] range (prediction market probability).
        size: Number of contracts available at this price level.
    """

    model_config = {"frozen": True}

    price: float = Field(ge=0.0, le=1.0)
    size: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _validate_finite(self) -> OrderBookLevel:
        """Reject NaN and infinity."""
        if math.isnan(self.price) or math.isinf(self.price):
            raise ValueError(f"price must be finite, got {self.price}")
        if math.isnan(self.size) or math.isinf(self.size):
            raise ValueError(f"size must be finite, got {self.size}")
        return self

    @property
    def notional(self) -> float:
        """Total notional value at this level (price * size)."""
        return self.price * self.size


def _canonicalize_levels(
    levels: list[dict[str, float]] | tuple[dict[str, float], ...],
    descending: bool,
) -> tuple[OrderBookLevel, ...]:
    """Aggregate and sort price levels deterministically.

    Args:
        levels: Raw price/size pairs.
        descending: True for bids (highest first), False for asks (lowest first).

    Returns:
        Sorted, aggregated tuple of OrderBookLevel.
    """
    aggregated: dict[float, float] = defaultdict(float)
    for level in levels:
        aggregated[level["price"]] += level["size"]

    sorted_prices = sorted(aggregated.keys(), reverse=descending)
    return tuple(
        OrderBookLevel(price=p, size=s)
        for p, sorted_sizes in [(p, aggregated[p]) for p in sorted_prices]
        for s in [sorted_sizes]
    )


class OrderBookSnapshot(BaseModel):
    """Immutable snapshot of order book depth for a prediction market instrument.

    Bids are sorted descending by price (best bid first).
    Asks are sorted ascending by price (best ask first).
    Duplicate price levels are aggregated during construction.

    Attributes:
        provider: Provider identifier (e.g. "polymarket", "kalshi").
        provider_instrument_id: Provider's instrument identifier.
        source_timestamp: When the provider generated this snapshot (UTC).
        source_timestamp_missing: True when provider has no source timestamp.
        ingestion_timestamp: When our system received this snapshot (UTC).
        bids: Bid levels, sorted best-first (highest price).
        asks: Ask levels, sorted best-first (lowest price).
        schema_version: Schema version for forward compatibility.
    """

    model_config = {"frozen": True}

    provider: str
    provider_instrument_id: str
    source_timestamp: datetime
    ingestion_timestamp: datetime
    source_timestamp_missing: bool = False
    bids: tuple[OrderBookLevel, ...] = ()
    asks: tuple[OrderBookLevel, ...] = ()
    schema_version: str = "1"

    @model_validator(mode="before")
    @classmethod
    def _canonicalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Canonicalize bids/asks: aggregate duplicates and sort."""
        data = dict(data)

        raw_bids = data.get("bids") or []
        raw_asks = data.get("asks") or []

        # Convert raw dicts to list[dict[str, float]] if needed
        bid_dicts = [
            {"price": b.price, "size": b.size} if hasattr(b, "price") else b for b in raw_bids
        ]
        ask_dicts = [
            {"price": a.price, "size": a.size} if hasattr(a, "price") else a for a in raw_asks
        ]

        data["bids"] = _canonicalize_levels(bid_dicts, descending=True)
        data["asks"] = _canonicalize_levels(ask_dicts, descending=False)

        return data

    @model_validator(mode="after")
    def _validate_timestamps(self) -> OrderBookSnapshot:
        """Validate timestamp constraints."""
        if self.source_timestamp.tzinfo is None:
            raise ValueError("source_timestamp must be timezone-aware")
        if self.ingestion_timestamp.tzinfo is None:
            raise ValueError("ingestion_timestamp must be timezone-aware")
        return self

    @property
    def best_bid_price(self) -> float | None:
        """Best bid price (highest). None if no bids."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask_price(self) -> float | None:
        """Best ask price (lowest). None if no asks."""
        return self.asks[0].price if self.asks else None

    @property
    def midpoint(self) -> float | None:
        """Midpoint price. None if either side is empty."""
        if self.bids and self.asks:
            return (self.bids[0].price + self.asks[0].price) / 2.0
        return None

    @property
    def total_bid_size(self) -> float:
        """Total size across all bid levels."""
        return sum(level.size for level in self.bids)

    @property
    def total_ask_size(self) -> float:
        """Total size across all ask levels."""
        return sum(level.size for level in self.asks)


def snapshot_id(snapshot: OrderBookSnapshot) -> str:
    """Compute deterministic snapshot identity.

    Uses SHA-256 of canonical JSON representation.
    Same logical content always produces the same ID.

    Canonical representation includes:
    - provider
    - provider_instrument_id
    - source_timestamp (excluded when source_timestamp_missing=True)
    - ingestion_timestamp
    - bids (sorted by price descending)
    - asks (sorted by price ascending)
    - schema_version

    Args:
        snapshot: The order book snapshot.

    Returns:
        16-character hexadecimal string.
    """
    d: dict[str, Any] = {
        "provider": snapshot.provider,
        "provider_instrument_id": snapshot.provider_instrument_id,
        "ingestion_timestamp": snapshot.ingestion_timestamp.isoformat(),
        "bids": [{"price": level.price, "size": level.size} for level in snapshot.bids],
        "asks": [{"price": level.price, "size": level.size} for level in snapshot.asks],
        "schema_version": snapshot.schema_version,
    }
    if not snapshot.source_timestamp_missing:
        d["source_timestamp"] = snapshot.source_timestamp.isoformat()

    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    import hashlib

    h = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return h[:16]
