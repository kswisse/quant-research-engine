"""Kalshi → canonical MarketQuote normalizer.

Converts Kalshi-specific raw order-book data into provider-independent
canonical MarketQuote records.

Architecture:
    Kalshi GetMarketOrderbookResponse
        ↓
    KalshiNormalizer.normalize_orderbook()
        ↓
    MarketQuote (top-of-book)

Kalshi order book semantics (critical):
    The Kalshi API only returns BIDS on each side. There are NO asks.
    - yes_dollars: array of YES bid levels
    - no_dollars: array of NO bid levels

    In a binary market, duality applies:
    - A YES bid at price X is economically equivalent to a NO ask at (1 - X)
    - Therefore: YES ask = 1.0 - best NO bid price
    - And: NO ask = 1.0 - best YES bid price

Top-of-book extraction:
    - Best YES bid = highest price in yes_dollars (last element, sorted ascending)
    - Best YES ask = 1.0 - best NO bid price (from no_dollars)
    - Bid size = count at the best YES bid level
    - Ask size = count at the best NO bid level (same contracts, duality)

Edge cases:
    - Empty order book: both bid and ask are None
    - YES bids only (no NO bids): bid exists, ask is None
    - NO bids only (no YES bids): bid is None, ask is None
      (since we quote on the YES side, a missing YES bid means no bid)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from quant_engine.market_data.errors import NormalizationError
from quant_engine.market_data.models import MarketQuote
from quant_engine.market_data.normalization import MarketDataNormalizer
from quant_engine.providers.kalshi.models import GetMarketOrderbookResponse


class KalshiNormalizer(MarketDataNormalizer):
    """Normalizer for Kalshi prediction market data.

    Converts Kalshi orderbook responses into canonical MarketQuote records
    suitable for the research engine.

    Example:
        ```python
        normalizer = KalshiNormalizer()
        quote = normalizer.normalize_orderbook(
            ticker="KXHIGHNY-24JAN01-T60",
            orderbook=response,
            source_timestamp=datetime.now(UTC),
        )
        ```
    """

    PROVIDER = "kalshi"

    def __init__(
        self,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Initialize the normalizer.

        Args:
            clock: Optional clock function for deterministic timestamps.
                   If None, uses datetime.now(UTC). Useful for testing.
        """
        self._clock = clock or (lambda: datetime.now(UTC))

    def _parse_timestamp(self, iso_str: str) -> datetime:
        """Parse ISO 8601 timestamp string to UTC datetime.

        Args:
            iso_str: ISO 8601 datetime string (e.g., "2024-01-15T10:30:00Z").

        Returns:
            Timezone-aware UTC datetime.

        Raises:
            NormalizationError: If the timestamp string is invalid.
        """
        try:
            dt = datetime.fromisoformat(iso_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt
        except (ValueError, TypeError) as exc:
            raise NormalizationError(f"Invalid ISO timestamp: {iso_str!r}") from exc

    def _extract_top_of_book(
        self,
        orderbook: GetMarketOrderbookResponse,
    ) -> tuple[float | None, float | None, float | None, float | None]:
        """Extract best bid/ask from a Kalshi order book.

        Kalshi only returns bids. The ask side is derived via duality:
            YES ask = 1.0 - best NO bid price

        Args:
            orderbook: The raw Kalshi orderbook response.

        Returns:
            Tuple of (bid_price, bid_size, ask_price, ask_size).
            Missing sides are None.
        """
        fp = orderbook.orderbook_fp

        # Best YES bid = highest price in yes_dollars (last element, ascending sort)
        bid_price: float | None = None
        bid_size: float | None = None
        if fp.yes_dollars:
            # Find the level with the highest price
            best_yes_bid = max(fp.yes_dollars, key=lambda lv: float(lv[0]))
            bid_price = float(best_yes_bid[0])
            bid_size = float(best_yes_bid[1])

        # Best YES ask = 1.0 - best NO bid price (binary duality)
        ask_price: float | None = None
        ask_size: float | None = None
        if fp.no_dollars:
            best_no_bid = max(fp.no_dollars, key=lambda lv: float(lv[0]))
            no_bid_price = float(best_no_bid[0])
            ask_price = round(1.0 - no_bid_price, 6)
            ask_size = float(best_no_bid[1])

        return bid_price, bid_size, ask_price, ask_size

    def normalize_orderbook(
        self,
        ticker: str,
        orderbook: GetMarketOrderbookResponse,
        source_timestamp: datetime | None = None,
        ingestion_timestamp: datetime | None = None,
    ) -> MarketQuote:
        """Normalize a Kalshi orderbook into a canonical MarketQuote.

        Extracts top-of-book (best bid and derived ask) from the order book.

        Args:
            ticker: The market ticker (used as provider_instrument_id).
            orderbook: The raw Kalshi orderbook response.
            source_timestamp: Override for source timestamp. If None,
                              uses ingestion_timestamp (Kalshi orderbook
                              has no timestamp field).
            ingestion_timestamp: Override for ingestion timestamp. If None,
                                 uses the clock.

        Returns:
            Canonical MarketQuote with best bid/ask.

        Raises:
            NormalizationError: If the orderbook is invalid or ticker
                               is empty.
        """
        # Determine timestamps
        if ingestion_timestamp is None:
            ingestion_timestamp = self._clock()

        # Kalshi orderbook has no timestamp; use ingestion time as source.
        # Mark source_timestamp_missing so downstream consumers know this
        # is NOT the actual exchange-side event time.
        source_timestamp_missing = source_timestamp is None
        if source_timestamp is None:
            source_timestamp = ingestion_timestamp

        # Validate ticker
        if not ticker or not ticker.strip():
            raise NormalizationError("ticker must be non-empty")

        # Extract top-of-book
        bid_price, bid_size, ask_price, ask_size = self._extract_top_of_book(orderbook)

        return MarketQuote(
            source_timestamp=source_timestamp,
            source_timestamp_missing=source_timestamp_missing,
            ingestion_timestamp=ingestion_timestamp,
            provider=self.PROVIDER,
            provider_instrument_id=ticker,
            bid_price=bid_price,
            bid_size=bid_size,
            ask_price=ask_price,
            ask_size=ask_size,
        )

    def normalize(self, raw_record: dict[str, Any]) -> MarketQuote:
        """Normalize a raw dict record into a canonical MarketQuote.

        This implements the MarketDataNormalizer ABC interface. It accepts
        a dict representation of an orderbook response and normalizes it.

        Expected dict keys:
            - orderbook_fp: { yes_dollars: [[price, count], ...],
                              no_dollars: [[price, count], ...] }

        Args:
            raw_record: Provider-specific raw data as a dictionary.

        Returns:
            Canonical MarketQuote.

        Raises:
            NormalizationError: If the raw record cannot be normalized.
        """
        try:
            orderbook = GetMarketOrderbookResponse(**raw_record)
        except Exception as exc:
            raise NormalizationError(f"Failed to parse raw record as orderbook: {exc}") from exc

        # For the dict interface, we need a ticker. Use a placeholder
        # that callers should override via normalize_orderbook().
        ticker = raw_record.get("ticker", "")

        return self.normalize_orderbook(
            ticker=ticker,
            orderbook=orderbook,
        )
