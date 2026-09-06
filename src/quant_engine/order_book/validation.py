"""Deterministic validation for order book snapshots.

Provides:
    validate_snapshot: Validate an OrderBookSnapshot
    validate_level: Validate a single OrderBookLevel
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quant_engine.order_book.models import OrderBookSnapshot


def validate_level(level: object) -> list[str]:
    """Validate a single order book level.

    Args:
        level: The level to validate.

    Returns:
        List of error strings. Empty list means valid.
    """
    errors: list[str] = []

    if not hasattr(level, "price") or not hasattr(level, "size"):
        errors.append("level must have price and size attributes")
        return errors

    price = level.price
    size = level.size

    if not isinstance(price, (int, float)):
        errors.append(f"price must be numeric, got {type(price).__name__}")
    elif price < 0.0 or price > 1.0:
        errors.append(f"price must be in [0.0, 1.0], got {price}")

    if not isinstance(size, (int, float)):
        errors.append(f"size must be numeric, got {type(size).__name__}")
    elif size < 0.0:
        errors.append(f"size must be >= 0.0, got {size}")

    return errors


def validate_snapshot(snapshot: OrderBookSnapshot) -> list[str]:
    """Validate an order book snapshot.

    Checks:
    - Timestamp semantics
    - Provider/instrument identifiers
    - Level values
    - Ordering invariants
    - Schema version
    - Crossed book (best_bid > best_ask)

    Does NOT reject crossed books at construction.
    Follows the existing MarketQuote validation pattern.

    Args:
        snapshot: The snapshot to validate.

    Returns:
        List of error strings. Empty list means valid.
    """
    errors: list[str] = []

    # Provider/instrument validation
    if not snapshot.provider or not snapshot.provider.strip():
        errors.append("provider must be non-empty")
    if not snapshot.provider_instrument_id or not snapshot.provider_instrument_id.strip():
        errors.append("provider_instrument_id must be non-empty")

    # Timestamp validation
    if snapshot.source_timestamp.tzinfo is None:
        errors.append("source_timestamp must be timezone-aware")
    if snapshot.ingestion_timestamp.tzinfo is None:
        errors.append("ingestion_timestamp must be timezone-aware")

    # Schema version validation
    if not snapshot.schema_version or not snapshot.schema_version.strip():
        errors.append("schema_version must be non-empty")

    # Validate individual levels
    for i, level in enumerate(snapshot.bids):
        level_errors = validate_level(level)
        for err in level_errors:
            errors.append(f"bids[{i}]: {err}")

    for i, level in enumerate(snapshot.asks):
        level_errors = validate_level(level)
        for err in level_errors:
            errors.append(f"asks[{i}]: {err}")

    # Ordering invariants
    for i in range(1, len(snapshot.bids)):
        if snapshot.bids[i].price > snapshot.bids[i - 1].price:
            errors.append(
                f"bids not sorted descending: bids[{i}].price ({snapshot.bids[i].price}) "
                f"> bids[{i - 1}].price ({snapshot.bids[i - 1].price})"
            )

    for i in range(1, len(snapshot.asks)):
        if snapshot.asks[i].price < snapshot.asks[i - 1].price:
            errors.append(
                f"asks not sorted ascending: asks[{i}].price ({snapshot.asks[i].price}) "
                f"< asks[{i - 1}].price ({snapshot.asks[i - 1].price})"
            )

    # Crossed book check
    if snapshot.bids and snapshot.asks:
        best_bid = snapshot.bids[0].price
        best_ask = snapshot.asks[0].price
        if best_bid > best_ask:
            errors.append(f"crossed book: best_bid ({best_bid}) > best_ask ({best_ask})")

    return errors
