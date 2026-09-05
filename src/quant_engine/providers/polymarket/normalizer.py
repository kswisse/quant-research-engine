"""Polymarket → canonical MarketQuote normalizer.

Converts Polymarket-specific raw order-book data into provider-independent
canonical MarketQuote records.

Architecture:
    PolymarketRawOrderBook / OrderBookSummary
        ↓
    PolymarketNormalizer.normalize_order_book()
        ↓
    MarketQuote (top-of-book)

Top-of-book extraction:
    - Best bid = highest price among all bid levels
    - Best ask = lowest price among all ask levels
    - Bid size = aggregate size at the best bid price level
    - Ask size = aggregate size at the best ask price level

Edge cases:
    - Empty order book: both bid and ask are None
    - One-sided book (bids only or asks only): missing side is None
    - Multiple levels at same price: sizes are summed
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from quant_engine.market_data.errors import NormalizationError
from quant_engine.market_data.models import MarketQuote
from quant_engine.market_data.normalization import MarketDataNormalizer
from quant_engine.providers.polymarket.models import OrderBookSummary


class PolymarketNormalizer(MarketDataNormalizer):
    """Normalizer for Polymarket prediction market data.

    Converts Polymarket OrderBookSummary and raw dict records into
    canonical MarketQuote records suitable for the research engine.

    Example:
        ```python
        normalizer = PolymarketNormalizer()
        quote = normalizer.normalize_order_book(
            token_id="12345...",
            order_book=book_summary,
            source_timestamp=datetime.now(UTC),
        )
        ```
    """

    PROVIDER = "polymarket"

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

    def _parse_timestamp(self, epoch_ms_str: str) -> datetime:
        """Convert epoch milliseconds string to UTC datetime.

        Args:
            epoch_ms_str: Epoch time in milliseconds as a string (e.g., "1782753357257").

        Returns:
            Timezone-aware UTC datetime.

        Raises:
            NormalizationError: If the timestamp string is invalid.
        """
        try:
            epoch_ms = int(epoch_ms_str)
            # Convert milliseconds to seconds
            epoch_s = epoch_ms / 1000.0
            return datetime.fromtimestamp(epoch_s, tz=UTC)
        except (ValueError, OSError) as exc:
            raise NormalizationError(
                f"Invalid epoch timestamp: {epoch_ms_str!r}"
            ) from exc

    def _extract_top_of_book(
        self,
        order_book: OrderBookSummary,
    ) -> tuple[float | None, float | None, float | None, float | None]:
        """Extract best bid/ask from an order book.

        The Polymarket CLOB API returns:
        - bids: sorted by price ascending (best bid is last)
        - asks: sorted by price descending (best ask is last)

        However, to be robust against sorting anomalies, we scan all levels
        and take the max bid / min ask.

        Args:
            order_book: The raw order book summary.

        Returns:
            Tuple of (bid_price, bid_size, ask_price, ask_size).
            Missing sides are None.
        """
        # Aggregate sizes at each price level (handles potential duplicates)
        bid_aggregate: dict[str, float] = defaultdict(float)
        ask_aggregate: dict[str, float] = defaultdict(float)

        for level in order_book.bids:
            bid_aggregate[level.price] += float(level.size)

        for level in order_book.asks:
            ask_aggregate[level.price] += float(level.size)

        # Best bid = highest bid price
        bid_price: float | None = None
        bid_size: float | None = None
        if bid_aggregate:
            best_bid_price_str = max(bid_aggregate.keys(), key=lambda p: float(p))
            bid_price = float(best_bid_price_str)
            bid_size = bid_aggregate[best_bid_price_str]

        # Best ask = lowest ask price
        ask_price: float | None = None
        ask_size: float | None = None
        if ask_aggregate:
            best_ask_price_str = min(ask_aggregate.keys(), key=lambda p: float(p))
            ask_price = float(best_ask_price_str)
            ask_size = ask_aggregate[best_ask_price_str]

        return bid_price, bid_size, ask_price, ask_size

    def normalize_order_book(
        self,
        token_id: str,
        order_book: OrderBookSummary,
        source_timestamp: datetime | None = None,
        ingestion_timestamp: datetime | None = None,
    ) -> MarketQuote:
        """Normalize an OrderBookSummary into a canonical MarketQuote.

        Extracts top-of-book (best bid and best ask) from the full order book.

        Args:
            token_id: The outcome token ID (used as provider_instrument_id).
            order_book: The raw Polymarket order book.
            source_timestamp: Override for source timestamp. If None, parsed
                              from order_book.timestamp (epoch ms string).
            ingestion_timestamp: Override for ingestion timestamp. If None,
                                 uses the clock.

        Returns:
            Canonical MarketQuote with best bid/ask.

        Raises:
            NormalizationError: If the order book is invalid or timestamps
                               cannot be parsed.
        """
        # Determine source timestamp
        if source_timestamp is None:
            source_timestamp = self._parse_timestamp(order_book.timestamp)

        # Determine ingestion timestamp
        if ingestion_timestamp is None:
            ingestion_timestamp = self._clock()

        # Validate token_id
        if not token_id or not token_id.strip():
            raise NormalizationError("token_id must be non-empty")

        # Extract top-of-book
        bid_price, bid_size, ask_price, ask_size = self._extract_top_of_book(
            order_book
        )

        return MarketQuote(
            source_timestamp=source_timestamp,
            ingestion_timestamp=ingestion_timestamp,
            provider=self.PROVIDER,
            provider_instrument_id=token_id,
            bid_price=bid_price,
            bid_size=bid_size,
            ask_price=ask_price,
            ask_size=ask_size,
        )

    def normalize(self, raw_record: dict[str, Any]) -> MarketQuote:
        """Normalize a raw dict record into a canonical MarketQuote.

        This implements the MarketDataNormalizer ABC interface. It accepts
        a dict representation of an OrderBookSummary (as returned by
        client.get_order_book()) and normalizes it.

        Expected dict keys:
            - market: Condition ID (hex string)
            - asset_id: Token ID (string)
            - timestamp: Epoch milliseconds as string
            - hash: Order book state hash
            - bids: List of {"price": str, "size": str} dicts
            - asks: List of {"price": str, "size": str} dicts
            - min_order_size, tick_size, neg_risk, last_trade_price

        Args:
            raw_record: Provider-specific raw data as a dictionary.

        Returns:
            Canonical MarketQuote.

        Raises:
            NormalizationError: If the raw record cannot be normalized.
        """
        try:
            # Parse into OrderBookSummary
            order_book = OrderBookSummary(**raw_record)
        except Exception as exc:
            raise NormalizationError(
                f"Failed to parse raw record as OrderBookSummary: {exc}"
            ) from exc

        return self.normalize_order_book(
            token_id=order_book.asset_id,
            order_book=order_book,
            source_timestamp=self._parse_timestamp(order_book.timestamp),
            ingestion_timestamp=self._clock(),
        )
