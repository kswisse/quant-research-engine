"""Deterministic validation for canonical market data records.

Provides:
    validate_quote: Validate a single MarketQuote
    validate_quotes: Validate a list of MarketQuote records
    validate_time_order: Detect out-of-order timestamps in a sequence
    find_duplicates: Detect duplicate records by identity
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from quant_engine.market_data.models import MarketQuote


def validate_quote(quote: MarketQuote) -> list[str]:
    """Validate a single market quote.

    Checks:
    - bid_price >= 0 if not None
    - ask_price >= 0 if not None
    - bid_size >= 0 if not None
    - ask_size >= 0 if not None
    - bid_price <= ask_price if both present (reject crossed markets)
    - No NaN or infinity in any numeric field
    - source_timestamp is timezone-aware
    - ingestion_timestamp is timezone-aware
    - provider and provider_instrument_id are non-empty

    Args:
        quote: The market quote to validate.

    Returns:
        List of validation error messages. Empty list means valid.
    """
    errors: list[str] = []

    # Timestamp checks
    if quote.source_timestamp.tzinfo is None:
        errors.append("source_timestamp is naive (not timezone-aware)")
    if quote.ingestion_timestamp.tzinfo is None:
        errors.append("ingestion_timestamp is naive (not timezone-aware)")

    # Provider checks
    if not quote.provider:
        errors.append("provider is empty")
    if not quote.provider_instrument_id:
        errors.append("provider_instrument_id is empty")

    # Numeric field checks
    for field_name in ("bid_price", "ask_price", "bid_size", "ask_size"):
        value = getattr(quote, field_name)
        if value is not None:
            if math.isnan(value):
                errors.append(f"{field_name} is NaN")
            elif math.isinf(value):
                errors.append(f"{field_name} is infinity")
            elif value < 0:
                errors.append(f"{field_name} is negative ({value})")

    # Crossed market check
    if (
        quote.bid_price is not None
        and quote.ask_price is not None
        and quote.bid_price > quote.ask_price
    ):
        errors.append(f"crossed market: bid ({quote.bid_price}) > ask ({quote.ask_price})")

    return errors


def validate_quotes(quotes: list[MarketQuote]) -> dict[int, list[str]]:
    """Validate a list of market quotes.

    Args:
        quotes: List of quotes to validate.

    Returns:
        Dict mapping index to list of errors. Empty dict means all valid.
    """
    all_errors: dict[int, list[str]] = {}
    for i, q in enumerate(quotes):
        errors = validate_quote(q)
        if errors:
            all_errors[i] = errors
    return all_errors


def validate_time_order(quotes: list[MarketQuote]) -> list[dict[str, object]]:
    """Detect out-of-order source timestamps in a sequence.

    Checks that source_timestamp is non-decreasing across consecutive records.
    Does NOT reorder — only reports violations.

    When source_timestamp_missing=True, the record is excluded from
    time-order checking because its source_timestamp is not a meaningful
    exchange-side event time.

    Args:
        quotes: Ordered list of quotes to check.

    Returns:
        List of violation dicts, each containing:
        - "index": index of the out-of-order record
        - "timestamp": the violating timestamp
        - "previous_timestamp": the previous timestamp
        Empty list means no violations.
    """
    violations: list[dict[str, object]] = []
    # Find the last non-missing timestamp for comparison
    last_valid_ts: datetime | None = None
    for i, q in enumerate(quotes):
        if q.source_timestamp_missing:
            continue
        if last_valid_ts is not None and q.source_timestamp < last_valid_ts:
            violations.append(
                {
                    "index": i,
                    "timestamp": q.source_timestamp.isoformat(),
                    "previous_timestamp": last_valid_ts.isoformat(),
                }
            )
        last_valid_ts = q.source_timestamp
    return violations


def find_duplicates(quotes: list[MarketQuote]) -> list[dict[str, object]]:
    """Detect duplicate records by canonical record identity.

    Two records are duplicates if they share the same record_id.
    Same record_id means same (provider, provider_instrument_id,
    source_timestamp, payload).

    Args:
        quotes: List of quotes to check.

    Returns:
        List of duplicate groups, each containing:
        - "record_id": the shared record_id
        - "indices": list of indices with this record_id
        Empty list means no duplicates.
    """
    seen: dict[str, list[int]] = {}
    for i, q in enumerate(quotes):
        rid = q.record_id
        if rid not in seen:
            seen[rid] = []
        seen[rid].append(i)

    duplicates: list[dict[str, object]] = []
    for rid, indices in seen.items():
        if len(indices) > 1:
            duplicates.append({"record_id": rid, "indices": indices})
    return duplicates
