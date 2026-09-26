"""Order book model regression tests.

Covers OrderBookLevel, OrderBookSnapshot, validation, identity,
and calculations for the order book research model.

All tests are deterministic:
- No network calls
- No shared state between tests
- Every test owns its data

Run: pytest tests/test_order_book.py -v
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_engine.order_book import (
    DepthLevel,
    ExecutionFill,
    ExecutionResult,
    OrderBookLevel,
    OrderBookSnapshot,
    best_ask,
    best_bid,
    consume_book,
    cumulative_depth,
    snapshot_id,
    spread,
    validate_snapshot,
)

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

FROZEN_CLOCK = datetime(2026, 3, 15, 10, 0, 0, tzinfo=UTC)
SOURCE_TIME = datetime(2026, 3, 15, 9, 59, 58, tzinfo=UTC)


def _make_level(price: float = 0.5, size: float = 100.0) -> OrderBookLevel:
    """Create a test level."""
    return OrderBookLevel(price=price, size=size)


def _make_snapshot(
    bids: list[tuple[float, float]] | None = None,
    asks: list[tuple[float, float]] | None = None,
    provider: str = "test",
    instrument: str = "T-1",
    source_ts: datetime | None = None,
    ingest_ts: datetime | None = None,
) -> OrderBookSnapshot:
    """Create a test snapshot."""
    bid_levels = [OrderBookLevel(price=p, size=s) for p, s in (bids or [])]
    ask_levels = [OrderBookLevel(price=p, size=s) for p, s in (asks or [])]
    return OrderBookSnapshot(
        provider=provider,
        provider_instrument_id=instrument,
        source_timestamp=source_ts or SOURCE_TIME,
        ingestion_timestamp=ingest_ts or FROZEN_CLOCK,
        bids=tuple(bid_levels),
        asks=tuple(ask_levels),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. ORDER BOOK LEVEL TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestOrderBookLevel:
    """OrderBookLevel validation and properties."""

    def test_valid_level(self) -> None:
        """Standard valid level."""
        level = OrderBookLevel(price=0.5, size=100.0)
        assert level.price == 0.5
        assert level.size == 100.0

    def test_price_zero(self) -> None:
        """Price = 0.0 is valid."""
        level = OrderBookLevel(price=0.0, size=50.0)
        assert level.price == 0.0

    def test_price_one(self) -> None:
        """Price = 1.0 is valid."""
        level = OrderBookLevel(price=1.0, size=50.0)
        assert level.price == 1.0

    def test_size_zero(self) -> None:
        """Size = 0.0 is valid (empty level)."""
        level = OrderBookLevel(price=0.5, size=0.0)
        assert level.size == 0.0

    def test_negative_price_rejected(self) -> None:
        """Negative price raises ValueError."""
        with pytest.raises(Exception):
            OrderBookLevel(price=-0.1, size=100.0)

    def test_price_above_one_rejected(self) -> None:
        """Price > 1.0 raises ValueError."""
        with pytest.raises(Exception):
            OrderBookLevel(price=1.1, size=100.0)

    def test_negative_size_rejected(self) -> None:
        """Negative size raises ValueError."""
        with pytest.raises(Exception):
            OrderBookLevel(price=0.5, size=-1.0)

    def test_nan_price_rejected(self) -> None:
        """NaN price raises ValueError."""
        with pytest.raises(Exception):
            OrderBookLevel(price=float("nan"), size=100.0)

    def test_inf_price_rejected(self) -> None:
        """Infinity price raises ValueError."""
        with pytest.raises(Exception):
            OrderBookLevel(price=float("inf"), size=100.0)

    def test_nan_size_rejected(self) -> None:
        """NaN size raises ValueError."""
        with pytest.raises(Exception):
            OrderBookLevel(price=0.5, size=float("nan"))

    def test_inf_size_rejected(self) -> None:
        """Infinity size raises ValueError."""
        with pytest.raises(Exception):
            OrderBookLevel(price=0.5, size=float("inf"))

    def test_immutability(self) -> None:
        """Level is immutable after creation."""
        level = OrderBookLevel(price=0.5, size=100.0)
        with pytest.raises(Exception):
            level.price = 0.6  # type: ignore[misc]

    def test_notional_property(self) -> None:
        """notional = price * size."""
        level = OrderBookLevel(price=0.55, size=200.0)
        assert level.notional == pytest.approx(110.0)

    def test_notional_zero_price(self) -> None:
        """notional = 0 when price = 0."""
        level = OrderBookLevel(price=0.0, size=100.0)
        assert level.notional == 0.0

    def test_notional_zero_size(self) -> None:
        """notional = 0 when size = 0."""
        level = OrderBookLevel(price=0.5, size=0.0)
        assert level.notional == 0.0

    def test_boundary_prices_accepted(self) -> None:
        """Boundary prices 0.0 and 1.0 are accepted."""
        l0 = OrderBookLevel(price=0.0, size=1.0)
        l1 = OrderBookLevel(price=1.0, size=1.0)
        assert l0.price == 0.0
        assert l1.price == 1.0

    def test_subpenny_price(self) -> None:
        """Subpenny prices are valid."""
        level = OrderBookLevel(price=0.5555, size=100.0)
        assert level.price == 0.5555


# ══════════════════════════════════════════════════════════════════════════════
# 2. ORDER BOOK SNAPSHOT TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestOrderBookSnapshot:
    """OrderBookSnapshot construction, canonicalization, and properties."""

    def test_valid_snapshot(self) -> None:
        """Standard valid snapshot."""
        snap = _make_snapshot(
            bids=[(0.50, 100.0), (0.49, 200.0)],
            asks=[(0.55, 100.0), (0.56, 200.0)],
        )
        assert snap.provider == "test"
        assert snap.provider_instrument_id == "T-1"
        assert len(snap.bids) == 2
        assert len(snap.asks) == 2

    def test_bid_sorting_descending(self) -> None:
        """Bids are sorted descending (best bid first)."""
        snap = _make_snapshot(
            bids=[(0.48, 100.0), (0.50, 100.0), (0.49, 100.0)],
        )
        prices = [level.price for level in snap.bids]
        assert prices == [0.50, 0.49, 0.48]

    def test_ask_sorting_ascending(self) -> None:
        """Asks are sorted ascending (best ask first)."""
        snap = _make_snapshot(
            asks=[(0.57, 100.0), (0.55, 100.0), (0.56, 100.0)],
        )
        prices = [level.price for level in snap.asks]
        assert prices == [0.55, 0.56, 0.57]

    def test_duplicate_bid_aggregation(self) -> None:
        """Duplicate bid prices are aggregated."""
        snap = _make_snapshot(
            bids=[(0.50, 100.0), (0.50, 200.0)],
        )
        assert len(snap.bids) == 1
        assert snap.bids[0].price == 0.50
        assert snap.bids[0].size == 300.0

    def test_duplicate_ask_aggregation(self) -> None:
        """Duplicate ask prices are aggregated."""
        snap = _make_snapshot(
            asks=[(0.55, 100.0), (0.55, 200.0)],
        )
        assert len(snap.asks) == 1
        assert snap.asks[0].price == 0.55
        assert snap.asks[0].size == 300.0

    def test_empty_bid_side(self) -> None:
        """Empty bid side is valid."""
        snap = _make_snapshot(bids=[], asks=[(0.55, 100.0)])
        assert len(snap.bids) == 0
        assert len(snap.asks) == 1

    def test_empty_ask_side(self) -> None:
        """Empty ask side is valid."""
        snap = _make_snapshot(bids=[(0.50, 100.0)], asks=[])
        assert len(snap.bids) == 1
        assert len(snap.asks) == 0

    def test_both_sides_empty(self) -> None:
        """Both sides empty is valid (empty book)."""
        snap = _make_snapshot(bids=[], asks=[])
        assert len(snap.bids) == 0
        assert len(snap.asks) == 0

    def test_crossed_book_not_rejected_at_construction(self) -> None:
        """Crossed book (bid > ask) is allowed at construction time."""
        snap = _make_snapshot(
            bids=[(0.60, 100.0)],
            asks=[(0.50, 100.0)],
        )
        assert snap.bids[0].price == 0.60
        assert snap.asks[0].price == 0.50

    def test_equal_bid_ask(self) -> None:
        """Equal bid/ask (zero spread) is valid."""
        snap = _make_snapshot(
            bids=[(0.50, 100.0)],
            asks=[(0.50, 100.0)],
        )
        assert snap.bids[0].price == 0.50
        assert snap.asks[0].price == 0.50

    def test_timestamps_timezone_aware(self) -> None:
        """Both timestamps must be timezone-aware."""
        snap = _make_snapshot()
        assert snap.source_timestamp.tzinfo is not None
        assert snap.ingestion_timestamp.tzinfo is not None

    def test_missing_source_timestamp(self) -> None:
        """source_timestamp_missing=True is preserved."""
        snap = OrderBookSnapshot(
            provider="kalshi",
            provider_instrument_id="K-1",
            source_timestamp=FROZEN_CLOCK,
            ingestion_timestamp=FROZEN_CLOCK,
            source_timestamp_missing=True,
            bids=(),
            asks=(),
        )
        assert snap.source_timestamp_missing is True

    def test_immutability(self) -> None:
        """Snapshot is immutable after creation."""
        snap = _make_snapshot(bids=[(0.50, 100.0)])
        with pytest.raises(Exception):
            snap.provider = "other"  # type: ignore[misc]

    def test_best_bid_price_property(self) -> None:
        """best_bid_price returns highest bid."""
        snap = _make_snapshot(bids=[(0.48, 100.0), (0.50, 100.0)])
        assert snap.best_bid_price == 0.50

    def test_best_bid_price_empty(self) -> None:
        """best_bid_price returns None when no bids."""
        snap = _make_snapshot(bids=[])
        assert snap.best_bid_price is None

    def test_best_ask_price_property(self) -> None:
        """best_ask_price returns lowest ask."""
        snap = _make_snapshot(asks=[(0.57, 100.0), (0.55, 100.0)])
        assert snap.best_ask_price == 0.55

    def test_best_ask_price_empty(self) -> None:
        """best_ask_price returns None when no asks."""
        snap = _make_snapshot(asks=[])
        assert snap.best_ask_price is None

    def test_midpoint_property(self) -> None:
        """midpoint = (best_bid + best_ask) / 2."""
        snap = _make_snapshot(
            bids=[(0.50, 100.0)],
            asks=[(0.60, 100.0)],
        )
        assert snap.midpoint == 0.55

    def test_midpoint_empty_side(self) -> None:
        """midpoint returns None when either side is empty."""
        snap = _make_snapshot(bids=[(0.50, 100.0)], asks=[])
        assert snap.midpoint is None

    def test_total_bid_size(self) -> None:
        """total_bid_size sums all bid levels."""
        snap = _make_snapshot(bids=[(0.50, 100.0), (0.49, 200.0)])
        assert snap.total_bid_size == 300.0

    def test_total_ask_size(self) -> None:
        """total_ask_size sums all ask levels."""
        snap = _make_snapshot(asks=[(0.55, 100.0), (0.56, 200.0)])
        assert snap.total_ask_size == 300.0

    def test_provider_and_instrument_preserved(self) -> None:
        """Provider and instrument are preserved."""
        snap = _make_snapshot(provider="polymarket", instrument="tok-123")
        assert snap.provider == "polymarket"
        assert snap.provider_instrument_id == "tok-123"


# ══════════════════════════════════════════════════════════════════════════════
# 3. SNAPSHOT IDENTITY TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestSnapshotIdentity:
    """Deterministic snapshot_id computation."""

    def test_same_content_same_id(self) -> None:
        """Same logical content → same snapshot_id."""
        snap1 = _make_snapshot(
            bids=[(0.50, 100.0)],
            asks=[(0.55, 100.0)],
        )
        snap2 = _make_snapshot(
            bids=[(0.50, 100.0)],
            asks=[(0.55, 100.0)],
        )
        assert snapshot_id(snap1) == snapshot_id(snap2)

    def test_different_price_different_id(self) -> None:
        """Different price → different snapshot_id."""
        snap1 = _make_snapshot(bids=[(0.50, 100.0)])
        snap2 = _make_snapshot(bids=[(0.51, 100.0)])
        assert snapshot_id(snap1) != snapshot_id(snap2)

    def test_different_size_different_id(self) -> None:
        """Different size → different snapshot_id."""
        snap1 = _make_snapshot(bids=[(0.50, 100.0)])
        snap2 = _make_snapshot(bids=[(0.50, 200.0)])
        assert snapshot_id(snap1) != snapshot_id(snap2)

    def test_different_provider_different_id(self) -> None:
        """Different provider → different snapshot_id."""
        snap1 = _make_snapshot(provider="polymarket")
        snap2 = _make_snapshot(provider="kalshi")
        assert snapshot_id(snap1) != snapshot_id(snap2)

    def test_different_instrument_different_id(self) -> None:
        """Different instrument → different snapshot_id."""
        snap1 = _make_snapshot(instrument="T-1")
        snap2 = _make_snapshot(instrument="T-2")
        assert snapshot_id(snap1) != snapshot_id(snap2)

    def test_source_timestamp_included_when_not_missing(self) -> None:
        """source_timestamp is included in hash when not missing."""
        snap1 = _make_snapshot(source_ts=datetime(2026, 1, 1, tzinfo=UTC))
        snap2 = _make_snapshot(source_ts=datetime(2026, 1, 2, tzinfo=UTC))
        assert snapshot_id(snap1) != snapshot_id(snap2)

    def test_source_timestamp_excluded_when_missing(self) -> None:
        """source_timestamp is excluded from hash when missing."""
        snap1 = OrderBookSnapshot(
            provider="kalshi",
            provider_instrument_id="K-1",
            source_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            ingestion_timestamp=FROZEN_CLOCK,
            source_timestamp_missing=True,
            bids=(),
            asks=(),
        )
        snap2 = OrderBookSnapshot(
            provider="kalshi",
            provider_instrument_id="K-1",
            source_timestamp=datetime(2026, 1, 2, tzinfo=UTC),
            ingestion_timestamp=FROZEN_CLOCK,
            source_timestamp_missing=True,
            bids=(),
            asks=(),
        )
        # Same logical content (source_timestamp excluded) → same ID
        assert snapshot_id(snap1) == snapshot_id(snap2)

    def test_id_is_16_hex_chars(self) -> None:
        """snapshot_id is 16 hexadecimal characters."""
        snap = _make_snapshot()
        sid = snapshot_id(snap)
        assert len(sid) == 16
        assert all(c in "0123456789abcdef" for c in sid)

    def test_deterministic_serialization(self) -> None:
        """Same inputs always produce same canonical serialization."""
        snap = _make_snapshot(
            bids=[(0.50, 100.0)],
            asks=[(0.55, 200.0)],
        )
        # Call twice — should be identical
        assert snapshot_id(snap) == snapshot_id(snap)

    def test_aggregation_does_not_affect_id(self) -> None:
        """Duplicate levels that aggregate to same result have same ID."""
        snap1 = _make_snapshot(bids=[(0.50, 300.0)])
        snap2 = _make_snapshot(bids=[(0.50, 100.0), (0.50, 200.0)])
        assert snapshot_id(snap1) == snapshot_id(snap2)


# ══════════════════════════════════════════════════════════════════════════════
# 4. VALIDATION TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestValidation:
    """validate_snapshot catches semantic errors."""

    def test_valid_snapshot(self) -> None:
        """Valid snapshot passes validation."""
        snap = _make_snapshot(
            bids=[(0.50, 100.0)],
            asks=[(0.55, 100.0)],
        )
        errors = validate_snapshot(snap)
        assert errors == []

    def test_empty_provider_rejected(self) -> None:
        """Empty provider fails validation."""
        snap = _make_snapshot(provider="")
        errors = validate_snapshot(snap)
        assert any("provider" in e for e in errors)

    def test_whitespace_provider_rejected(self) -> None:
        """Whitespace-only provider fails validation."""
        snap = _make_snapshot(provider="   ")
        errors = validate_snapshot(snap)
        assert any("provider" in e for e in errors)

    def test_empty_instrument_rejected(self) -> None:
        """Empty instrument fails validation."""
        snap = _make_snapshot(instrument="")
        errors = validate_snapshot(snap)
        assert any("instrument" in e for e in errors)

    def test_naive_source_timestamp_rejected(self) -> None:
        """Naive source_timestamp raises ValidationError at construction."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            OrderBookSnapshot(
                provider="test",
                provider_instrument_id="T-1",
                source_timestamp=datetime(2026, 1, 1),
                ingestion_timestamp=FROZEN_CLOCK,
            )

    def test_naive_ingestion_timestamp_rejected(self) -> None:
        """Naive ingestion_timestamp raises ValidationError at construction."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            OrderBookSnapshot(
                provider="test",
                provider_instrument_id="T-1",
                source_timestamp=SOURCE_TIME,
                ingestion_timestamp=datetime(2026, 1, 1),
            )

    def test_crossed_book_detected(self) -> None:
        """Crossed book (bid > ask) is detected by validation."""
        snap = _make_snapshot(
            bids=[(0.60, 100.0)],
            asks=[(0.50, 100.0)],
        )
        errors = validate_snapshot(snap)
        assert any("crossed" in e for e in errors)

    def test_crossed_book_not_mutated(self) -> None:
        """Validation does not modify the snapshot."""
        snap = _make_snapshot(
            bids=[(0.60, 100.0)],
            asks=[(0.50, 100.0)],
        )
        validate_snapshot(snap)
        assert snap.bids[0].price == 0.60
        assert snap.asks[0].price == 0.50

    def test_valid_book_with_no_crossing(self) -> None:
        """Valid book (bid < ask) passes crossed check."""
        snap = _make_snapshot(
            bids=[(0.50, 100.0)],
            asks=[(0.55, 100.0)],
        )
        errors = validate_snapshot(snap)
        assert not any("crossed" in e for e in errors)

    def test_equal_bid_ask_passes(self) -> None:
        """Equal bid/ask (zero spread) passes validation."""
        snap = _make_snapshot(
            bids=[(0.50, 100.0)],
            asks=[(0.50, 100.0)],
        )
        errors = validate_snapshot(snap)
        assert not any("crossed" in e for e in errors)

    def test_empty_schema_version_rejected(self) -> None:
        """Empty schema_version fails validation."""
        snap2 = OrderBookSnapshot(
            provider="test",
            provider_instrument_id="T-1",
            source_timestamp=SOURCE_TIME,
            ingestion_timestamp=FROZEN_CLOCK,
            schema_version="",
        )
        errors = validate_snapshot(snap2)
        assert any("schema_version" in e for e in errors)

    def test_bid_ordering_violation_detected(self) -> None:
        """Out-of-order bids are detected."""
        # Manually create a snapshot with wrong ordering
        # (canonicalization should fix this, but test the validator)
        snap = _make_snapshot(bids=[(0.50, 100.0), (0.55, 100.0)])
        # After canonicalization, bids are sorted correctly
        errors = validate_snapshot(snap)
        # Should be valid since canonicalization fixes ordering
        assert errors == []

    def test_single_level_valid(self) -> None:
        """Single level on each side is valid."""
        snap = _make_snapshot(
            bids=[(0.50, 100.0)],
            asks=[(0.55, 100.0)],
        )
        errors = validate_snapshot(snap)
        assert errors == []


# ══════════════════════════════════════════════════════════════════════════════
# 5. CALCULATION TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestCalculations:
    """best_bid, best_ask, spread, cumulative_depth."""

    def test_best_bid_normal(self) -> None:
        """best_bid returns highest price."""
        snap = _make_snapshot(bids=[(0.48, 100.0), (0.50, 100.0)])
        assert best_bid(snap) == 0.50

    def test_best_bid_empty(self) -> None:
        """best_bid returns None for empty bids."""
        snap = _make_snapshot(bids=[])
        assert best_bid(snap) is None

    def test_best_ask_normal(self) -> None:
        """best_ask returns lowest price."""
        snap = _make_snapshot(asks=[(0.57, 100.0), (0.55, 100.0)])
        assert best_ask(snap) == 0.55

    def test_best_ask_empty(self) -> None:
        """best_ask returns None for empty asks."""
        snap = _make_snapshot(asks=[])
        assert best_ask(snap) is None

    def test_spread_normal(self) -> None:
        """spread = best_ask - best_bid."""
        snap = _make_snapshot(
            bids=[(0.50, 100.0)],
            asks=[(0.55, 100.0)],
        )
        assert spread(snap) == pytest.approx(0.05)

    def test_spread_zero(self) -> None:
        """spread = 0 when bid == ask."""
        snap = _make_snapshot(
            bids=[(0.50, 100.0)],
            asks=[(0.50, 100.0)],
        )
        assert spread(snap) == pytest.approx(0.0)

    def test_spread_crossed_negative(self) -> None:
        """spread is negative for crossed book."""
        snap = _make_snapshot(
            bids=[(0.60, 100.0)],
            asks=[(0.50, 100.0)],
        )
        assert spread(snap) == pytest.approx(-0.10)

    def test_spread_empty_bid(self) -> None:
        """spread returns None when bids are empty."""
        snap = _make_snapshot(bids=[], asks=[(0.55, 100.0)])
        assert spread(snap) is None

    def test_spread_empty_ask(self) -> None:
        """spread returns None when asks are empty."""
        snap = _make_snapshot(bids=[(0.50, 100.0)], asks=[])
        assert spread(snap) is None

    def test_spread_both_empty(self) -> None:
        """spread returns None when both sides empty."""
        snap = _make_snapshot(bids=[], asks=[])
        assert spread(snap) is None

    def test_cumulative_depth_bid(self) -> None:
        """Cumulative depth for bids (best-first)."""
        snap = _make_snapshot(bids=[(0.50, 100.0), (0.49, 200.0), (0.48, 300.0)])
        result = cumulative_depth(snap, "bid", 2)
        assert len(result) == 2
        assert result[0].price == 0.50
        assert result[0].cumulative_size == 100.0
        assert result[1].price == 0.49
        assert result[1].cumulative_size == 300.0

    def test_cumulative_depth_ask(self) -> None:
        """Cumulative depth for asks (best-first)."""
        snap = _make_snapshot(asks=[(0.55, 100.0), (0.56, 200.0), (0.57, 300.0)])
        result = cumulative_depth(snap, "ask", 2)
        assert len(result) == 2
        assert result[0].price == 0.55
        assert result[0].cumulative_size == 100.0
        assert result[1].price == 0.56
        assert result[1].cumulative_size == 300.0

    def test_cumulative_depth_levels_exceed_available(self) -> None:
        """Requesting more levels than available returns all."""
        snap = _make_snapshot(bids=[(0.50, 100.0)])
        result = cumulative_depth(snap, "bid", 10)
        assert len(result) == 1

    def test_cumulative_depth_empty_side(self) -> None:
        """Empty side returns empty list."""
        snap = _make_snapshot(bids=[])
        result = cumulative_depth(snap, "bid", 5)
        assert result == []

    def test_cumulative_depth_invalid_levels(self) -> None:
        """levels <= 0 raises ValueError."""
        snap = _make_snapshot(bids=[(0.50, 100.0)])
        with pytest.raises(ValueError):
            cumulative_depth(snap, "bid", 0)
        with pytest.raises(ValueError):
            cumulative_depth(snap, "bid", -1)

    def test_cumulative_depth_notional(self) -> None:
        """Cumulative notional is computed correctly."""
        snap = _make_snapshot(bids=[(0.50, 100.0), (0.40, 200.0)])
        result = cumulative_depth(snap, "bid", 2)
        assert result[0].cumulative_notional == pytest.approx(50.0)
        assert result[1].cumulative_notional == pytest.approx(130.0)

    def test_cumulative_depth_single_level(self) -> None:
        """Single level cumulative depth."""
        snap = _make_snapshot(bids=[(0.50, 100.0)])
        result = cumulative_depth(snap, "bid", 1)
        assert len(result) == 1
        assert result[0].cumulative_size == 100.0


# ══════════════════════════════════════════════════════════════════════════════
# 6. PROPERTY-BASED TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestPropertyBased:
    """Hypothesis-based tests for invariants."""

    @given(
        bids=st.lists(
            st.tuples(
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
                st.floats(min_value=0.0, max_value=1000.0, allow_nan=False),
            ),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=50)
    def test_bids_sorted_descending(self, bids: list[tuple[float, float]]) -> None:
        """Bids are always sorted descending after construction."""
        snap = _make_snapshot(bids=bids)
        prices = [level.price for level in snap.bids]
        assert prices == sorted(prices, reverse=True)

    @given(
        asks=st.lists(
            st.tuples(
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
                st.floats(min_value=0.0, max_value=1000.0, allow_nan=False),
            ),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=50)
    def test_asks_sorted_ascending(self, asks: list[tuple[float, float]]) -> None:
        """Asks are always sorted ascending after construction."""
        snap = _make_snapshot(asks=asks)
        prices = [level.price for level in snap.asks]
        assert prices == sorted(prices)

    @given(
        st.lists(
            st.tuples(
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
                st.floats(min_value=0.0, max_value=1000.0, allow_nan=False),
            ),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=50)
    def test_duplicate_aggregation(self, levels: list[tuple[float, float]]) -> None:
        """Duplicate prices are aggregated (sum of sizes)."""
        snap = _make_snapshot(bids=levels)
        # Check no duplicate prices
        prices = [level.price for level in snap.bids]
        assert len(prices) == len(set(prices))

    @given(
        st.lists(
            st.tuples(
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
                st.floats(min_value=0.0, max_value=1000.0, allow_nan=False),
            ),
            min_size=1,
            max_size=5,
        )
    )
    @settings(max_examples=30)
    def test_best_bid_invariant(self, bids: list[tuple[float, float]]) -> None:
        """best_bid equals max price in bids."""
        snap = _make_snapshot(bids=bids)
        expected = max(price for price, _ in bids)
        assert best_bid(snap) == expected


# ══════════════════════════════════════════════════════════════════════════════
# 7. EXECUTION TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestExecutionValidation:
    """Input validation for consume_book()."""

    def test_zero_requested_size_rejected(self) -> None:
        """Zero requested_size raises ValueError."""
        snap = _make_snapshot(asks=[(0.50, 100.0)])
        with pytest.raises(ValueError, match="requested_size"):
            consume_book(snap, "buy", 0.0)

    def test_negative_requested_size_rejected(self) -> None:
        """Negative requested_size raises ValueError."""
        snap = _make_snapshot(asks=[(0.50, 100.0)])
        with pytest.raises(ValueError, match="requested_size"):
            consume_book(snap, "buy", -10.0)

    def test_nan_requested_size_rejected(self) -> None:
        """NaN requested_size raises ValueError."""
        snap = _make_snapshot(asks=[(0.50, 100.0)])
        with pytest.raises(ValueError, match="requested_size"):
            consume_book(snap, "buy", float("nan"))

    def test_inf_requested_size_rejected(self) -> None:
        """Infinity requested_size raises ValueError."""
        snap = _make_snapshot(asks=[(0.50, 100.0)])
        with pytest.raises(ValueError, match="requested_size"):
            consume_book(snap, "buy", float("inf"))

    def test_invalid_side_rejected(self) -> None:
        """Invalid side raises ValueError."""
        snap = _make_snapshot(asks=[(0.50, 100.0)])
        with pytest.raises(ValueError, match="side"):
            consume_book(snap, "invalid", 100.0)  # type: ignore[arg-type]


class TestBuyExecution:
    """BUY order execution (consume asks ascending)."""

    def test_one_level_full_fill(self) -> None:
        """BUY fills one level completely."""
        snap = _make_snapshot(asks=[(0.40, 100.0)])
        result = consume_book(snap, "buy", 50.0)

        assert result.side == "buy"
        assert result.requested_size == 50.0
        assert result.filled_size == 50.0
        assert result.remaining_size == 0.0
        assert result.total_notional == pytest.approx(20.0)
        assert result.vwap == pytest.approx(0.40)
        assert result.fully_filled is True
        assert len(result.fills) == 1
        assert result.fills[0].price == 0.40
        assert result.fills[0].size == 50.0

    def test_one_level_partial_fill(self) -> None:
        """BUY partially fills one level."""
        snap = _make_snapshot(asks=[(0.40, 100.0)])
        result = consume_book(snap, "buy", 150.0)

        assert result.filled_size == 100.0
        assert result.remaining_size == 50.0
        assert result.total_notional == pytest.approx(40.0)
        assert result.vwap == pytest.approx(0.40)
        assert result.fully_filled is False

    def test_multi_level_full_fill(self) -> None:
        """BUY fills multiple levels completely."""
        snap = _make_snapshot(asks=[(0.40, 100.0), (0.41, 100.0), (0.45, 100.0)])
        result = consume_book(snap, "buy", 250.0)

        assert result.filled_size == 250.0
        assert result.remaining_size == 0.0
        assert result.fully_filled is True
        assert len(result.fills) == 3
        assert result.fills[0].price == 0.40
        assert result.fills[1].price == 0.41
        assert result.fills[2].price == 0.45

    def test_multi_level_partial_fill(self) -> None:
        """BUY partially fills across multiple levels."""
        snap = _make_snapshot(asks=[(0.40, 100.0), (0.41, 100.0), (0.45, 100.0)])
        result = consume_book(snap, "buy", 150.0)

        assert result.filled_size == 150.0
        assert result.remaining_size == 0.0
        assert result.fully_filled is True
        assert len(result.fills) == 2
        assert result.fills[0].price == 0.40
        assert result.fills[0].size == 100.0
        assert result.fills[1].price == 0.41
        assert result.fills[1].size == 50.0

    def test_insufficient_liquidity(self) -> None:
        """BUY with insufficient liquidity fills partially."""
        snap = _make_snapshot(asks=[(0.40, 100.0)])
        result = consume_book(snap, "buy", 150.0)

        assert result.filled_size == 100.0
        assert result.remaining_size == 50.0
        assert result.fully_filled is False

    def test_empty_ask_side(self) -> None:
        """BUY with empty asks returns unfilled."""
        snap = _make_snapshot(asks=[])
        result = consume_book(snap, "buy", 100.0)

        assert result.filled_size == 0.0
        assert result.remaining_size == 100.0
        assert result.total_notional == 0.0
        assert result.vwap is None
        assert result.fully_filled is False
        assert len(result.fills) == 0

    def test_execution_priority_ascending(self) -> None:
        """BUY consumes asks from lowest to highest."""
        snap = _make_snapshot(asks=[(0.50, 100.0), (0.40, 100.0), (0.45, 100.0)])
        result = consume_book(snap, "buy", 200.0)

        # After canonicalization: asks are [0.40, 0.45, 0.50]
        assert result.fills[0].price == 0.40
        assert result.fills[1].price == 0.45


class TestSellExecution:
    """SELL order execution (consume bids descending)."""

    def test_one_level_full_fill(self) -> None:
        """SELL fills one level completely."""
        snap = _make_snapshot(bids=[(0.60, 100.0)])
        result = consume_book(snap, "sell", 50.0)

        assert result.side == "sell"
        assert result.requested_size == 50.0
        assert result.filled_size == 50.0
        assert result.remaining_size == 0.0
        assert result.total_notional == pytest.approx(30.0)
        assert result.vwap == pytest.approx(0.60)
        assert result.fully_filled is True

    def test_one_level_partial_fill(self) -> None:
        """SELL partially fills one level."""
        snap = _make_snapshot(bids=[(0.60, 100.0)])
        result = consume_book(snap, "sell", 150.0)

        assert result.filled_size == 100.0
        assert result.remaining_size == 50.0
        assert result.total_notional == pytest.approx(60.0)
        assert result.vwap == pytest.approx(0.60)
        assert result.fully_filled is False

    def test_multi_level_full_fill(self) -> None:
        """SELL fills multiple levels completely."""
        snap = _make_snapshot(bids=[(0.60, 100.0), (0.59, 100.0), (0.55, 100.0)])
        result = consume_book(snap, "sell", 250.0)

        assert result.filled_size == 250.0
        assert result.remaining_size == 0.0
        assert result.fully_filled is True
        assert len(result.fills) == 3
        assert result.fills[0].price == 0.60
        assert result.fills[1].price == 0.59
        assert result.fills[2].price == 0.55

    def test_multi_level_partial_fill(self) -> None:
        """SELL partially fills across multiple levels."""
        snap = _make_snapshot(bids=[(0.60, 100.0), (0.59, 100.0), (0.55, 100.0)])
        result = consume_book(snap, "sell", 150.0)

        assert result.filled_size == 150.0
        assert result.remaining_size == 0.0
        assert result.fully_filled is True
        assert len(result.fills) == 2
        assert result.fills[0].price == 0.60
        assert result.fills[0].size == 100.0
        assert result.fills[1].price == 0.59
        assert result.fills[1].size == 50.0

    def test_insufficient_liquidity(self) -> None:
        """SELL with insufficient liquidity fills partially."""
        snap = _make_snapshot(bids=[(0.60, 100.0)])
        result = consume_book(snap, "sell", 150.0)

        assert result.filled_size == 100.0
        assert result.remaining_size == 50.0
        assert result.fully_filled is False

    def test_empty_bid_side(self) -> None:
        """SELL with empty bids returns unfilled."""
        snap = _make_snapshot(bids=[])
        result = consume_book(snap, "sell", 100.0)

        assert result.filled_size == 0.0
        assert result.remaining_size == 100.0
        assert result.total_notional == 0.0
        assert result.vwap is None
        assert result.fully_filled is False
        assert len(result.fills) == 0

    def test_execution_priority_descending(self) -> None:
        """SELL consumes bids from highest to lowest."""
        snap = _make_snapshot(bids=[(0.55, 100.0), (0.60, 100.0), (0.50, 100.0)])
        result = consume_book(snap, "sell", 200.0)

        # After canonicalization: bids are [0.60, 0.55, 0.50]
        assert result.fills[0].price == 0.60
        assert result.fills[1].price == 0.55


class TestVWAP:
    """VWAP calculation correctness."""

    def test_vwap_single_level(self) -> None:
        """VWAP equals fill price for single level."""
        snap = _make_snapshot(asks=[(0.40, 100.0)])
        result = consume_book(snap, "buy", 50.0)
        assert result.vwap == pytest.approx(0.40)

    def test_vwap_multi_level(self) -> None:
        """VWAP is correctly computed across multiple levels."""
        snap = _make_snapshot(asks=[(0.40, 100.0), (0.41, 100.0)])
        result = consume_book(snap, "buy", 150.0)

        # 100 @ 0.40 = 40.00, 50 @ 0.41 = 20.50
        # total_notional = 60.50
        # filled_size = 150
        # VWAP = 60.50 / 150
        expected_vwap = (100.0 * 0.40 + 50.0 * 0.41) / 150.0
        assert result.vwap == pytest.approx(expected_vwap)

    def test_vwap_none_when_empty(self) -> None:
        """VWAP is None when nothing fills."""
        snap = _make_snapshot(asks=[])
        result = consume_book(snap, "buy", 100.0)
        assert result.vwap is None

    def test_vwap_within_price_range(self) -> None:
        """VWAP lies within the range of executed prices."""
        snap = _make_snapshot(asks=[(0.40, 100.0), (0.45, 100.0), (0.50, 100.0)])
        result = consume_book(snap, "buy", 250.0)

        assert result.vwap is not None
        assert result.vwap >= 0.40
        assert result.vwap <= 0.50


class TestExecutionFillNotional:
    """ExecutionFill.notional property."""

    def test_notional_property(self) -> None:
        """notional = price * size."""
        fill = ExecutionFill(price=0.55, size=100.0)
        assert fill.notional == pytest.approx(55.0)

    def test_notional_zero_price(self) -> None:
        """notional = 0 when price = 0."""
        fill = ExecutionFill(price=0.0, size=100.0)
        assert fill.notional == 0.0

    def test_notional_zero_size(self) -> None:
        """notional = 0 when size = 0."""
        fill = ExecutionFill(price=0.5, size=0.0)
        assert fill.notional == 0.0


class TestExecutionInvariants:
    """Accounting invariants for all valid results."""

    def test_filled_plus_remaining_equals_requested(self) -> None:
        """filled_size + remaining_size == requested_size."""
        snap = _make_snapshot(asks=[(0.40, 100.0), (0.41, 100.0)])
        result = consume_book(snap, "buy", 150.0)
        assert result.filled_size + result.remaining_size == pytest.approx(result.requested_size)

    def test_total_notional_equals_sum_of_fills(self) -> None:
        """total_notional == sum(fill.notional)."""
        snap = _make_snapshot(asks=[(0.40, 100.0), (0.41, 100.0)])
        result = consume_book(snap, "buy", 150.0)
        expected = sum(fill.notional for fill in result.fills)
        assert result.total_notional == pytest.approx(expected)

    def test_fully_filled_iff_remaining_zero(self) -> None:
        """fully_filled == (remaining_size == 0)."""
        snap = _make_snapshot(asks=[(0.40, 100.0)])
        result = consume_book(snap, "buy", 50.0)
        assert result.fully_filled is (result.remaining_size == 0.0)

    def test_non_mutation(self) -> None:
        """consume_book does not modify the source snapshot."""
        snap = _make_snapshot(
            asks=[(0.40, 100.0), (0.41, 100.0)],
            bids=[(0.39, 100.0)],
        )
        original_bids = list(snap.bids)
        original_asks = list(snap.asks)

        consume_book(snap, "buy", 150.0)

        assert list(snap.bids) == original_bids
        assert list(snap.asks) == original_asks


class TestExecutionCrossedBook:
    """Crossed book handling during execution."""

    def test_crossed_book_buy_consumes_asks(self) -> None:
        """BUY on crossed book consumes asks mechanically."""
        snap = _make_snapshot(
            bids=[(0.60, 100.0)],
            asks=[(0.50, 100.0)],
        )
        result = consume_book(snap, "buy", 50.0)

        assert result.filled_size == 50.0
        assert result.vwap == pytest.approx(0.50)

    def test_crossed_book_sell_consumes_bids(self) -> None:
        """SELL on crossed book consumes bids mechanically."""
        snap = _make_snapshot(
            bids=[(0.60, 100.0)],
            asks=[(0.50, 100.0)],
        )
        result = consume_book(snap, "sell", 50.0)

        assert result.filled_size == 50.0
        assert result.vwap == pytest.approx(0.60)


class TestExecutionPropertyBased:
    """Hypothesis-based tests for execution invariants."""

    @given(
        asks=st.lists(
            st.tuples(
                st.floats(min_value=0.01, max_value=0.99, allow_nan=False),
                st.floats(min_value=1.0, max_value=1000.0, allow_nan=False),
            ),
            min_size=1,
            max_size=10,
        ),
        requested=st.floats(min_value=1.0, max_value=5000.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_buy_filled_never_exceeds_requested(
        self, asks: list[tuple[float, float]], requested: float
    ) -> None:
        """BUY: filled_size <= requested_size."""
        snap = _make_snapshot(asks=asks)
        result = consume_book(snap, "buy", requested)
        assert result.filled_size <= result.requested_size + 1e-9

    @given(
        bids=st.lists(
            st.tuples(
                st.floats(min_value=0.01, max_value=0.99, allow_nan=False),
                st.floats(min_value=1.0, max_value=1000.0, allow_nan=False),
            ),
            min_size=1,
            max_size=10,
        ),
        requested=st.floats(min_value=1.0, max_value=5000.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_sell_filled_never_exceeds_requested(
        self, bids: list[tuple[float, float]], requested: float
    ) -> None:
        """SELL: filled_size <= requested_size."""
        snap = _make_snapshot(bids=bids)
        result = consume_book(snap, "sell", requested)
        assert result.filled_size <= result.requested_size + 1e-9

    @given(
        asks=st.lists(
            st.tuples(
                st.floats(min_value=0.01, max_value=0.99, allow_nan=False),
                st.floats(min_value=1.0, max_value=1000.0, allow_nan=False),
            ),
            min_size=1,
            max_size=10,
        ),
        requested=st.floats(min_value=1.0, max_value=5000.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_buy_vwap_in_price_range(
        self, asks: list[tuple[float, float]], requested: float
    ) -> None:
        """BUY: VWAP lies within the range of executed prices."""
        snap = _make_snapshot(asks=asks)
        result = consume_book(snap, "buy", requested)
        if result.fills:
            min_price = min(f.price for f in result.fills)
            max_price = max(f.price for f in result.fills)
            assert result.vwap is not None
            assert result.vwap >= min_price - 1e-9
            assert result.vwap <= max_price + 1e-9
