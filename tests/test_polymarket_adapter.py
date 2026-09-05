"""Tests for Polymarket market data adapter.

All tests use deterministic fixtures and mocked responses.
No live API calls. No network access required.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from quant_engine.market_data import (
    Dataset,
    InvalidTimestampError,
    MarketQuote,
)
from quant_engine.market_data.errors import NormalizationError
from quant_engine.market_data.validation import validate_quote
from quant_engine.providers.polymarket.client import (
    PolymarketAPIError,
    PolymarketClient,
    PolymarketClientError,
    PolymarketConnectionError,
    PolymarketTimeoutError,
)
from quant_engine.providers.polymarket.models import (
    ErrorResponse,
    EventMetadata,
    LastTradePriceResponse,
    MarketMetadata,
    MidpointResponse,
    OrderBookLevel,
    OrderBookSummary,
    PriceResponse,
    SpreadResponse,
)


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════


def _utc(iso: str) -> datetime:
    """Parse ISO string to UTC datetime."""
    return datetime.fromisoformat(iso)


# ── Raw API Response Fixtures ─────────────────────────────────────────────


SAMPLE_ORDER_BOOK_RAW: dict[str, Any] = {
    "market": "0x747dc2d490b14e4e5ee3c3e8d2ca7e6b3ee3a1a26c4b0e5e9c4c5c6e7f8a9b0",
    "asset_id": "12345678901234567890",
    "timestamp": "1782753357257",
    "hash": "a1b2c3d4e5f6",
    "bids": [
        {"price": "0.54", "size": "200.0", "timestamp": "1782753357257"},
        {"price": "0.55", "size": "100.0", "timestamp": "1782753357257"},
    ],
    "asks": [
        {"price": "0.59", "size": "100.0", "timestamp": "1782753357257"},
        {"price": "0.58", "size": "150.0", "timestamp": "1782753357257"},
    ],
    "min_order_size": "1.0",
    "tick_size": "0.01",
    "neg_risk": False,
    "last_trade_price": "0.56",
}

SAMPLE_MARKET_METADATA_RAW: dict[str, Any] = {
    "id": "703257",
    "slug": "will-bitcoin-hit-100k-2026",
    "question": "Will Bitcoin hit $100,000 in 2026?",
    "condition_id": "0x747dc2d490b14e4e5ee3c3e8d2ca7e6b3ee3a1a26c4b0e5e9c4c5c6e7f8a9b0",
    "clob_token_ids": '["12345678901234567890", "09876543210987654321"]',
}

SAMPLE_EVENT_METADATA_RAW: dict[str, Any] = {
    "id": "12345",
    "slug": "presidential-election-2028",
    "title": "2028 US Presidential Election",
    "markets": [SAMPLE_MARKET_METADATA_RAW],
}

SAMPLE_PRICE_RESPONSE_RAW: dict[str, Any] = {"price": "0.56"}

SAMPLE_MIDPOINT_RESPONSE_RAW: dict[str, Any] = {"mid": "0.565"}

SAMPLE_SPREAD_RESPONSE_RAW: dict[str, Any] = {"spread": "0.03"}

SAMPLE_LAST_TRADE_PRICE_RAW: dict[str, Any] = {"price": "0.56", "side": "BUY"}


# ── Parsed Model Fixtures ────────────────────────────────────────────────


@pytest.fixture
def sample_bid_levels() -> list[OrderBookLevel]:
    """Order book levels sorted by price ascending (best bid last)."""
    return [
        OrderBookLevel(price="0.54", size="200.0"),
        OrderBookLevel(price="0.55", size="100.0"),
    ]


@pytest.fixture
def sample_ask_levels() -> list[OrderBookLevel]:
    """Order book levels sorted by price descending (best ask last)."""
    return [
        OrderBookLevel(price="0.59", size="100.0"),
        OrderBookLevel(price="0.58", size="150.0"),
    ]


@pytest.fixture
def sample_order_book(
    sample_bid_levels: list[OrderBookLevel],
    sample_ask_levels: list[OrderBookLevel],
) -> OrderBookSummary:
    """A realistic OrderBookSummary with two levels on each side."""
    return OrderBookSummary(
        market="0x747dc2d490b14e4e5ee3c3e8d2ca7e6b3ee3a1a26c4b0e5e9c4c5c6e7f8a9b0",
        asset_id="12345678901234567890",
        timestamp="1782753357257",
        hash="a1b2c3d4e5f6",
        bids=sample_bid_levels,
        asks=sample_ask_levels,
        min_order_size="1.0",
        tick_size="0.01",
        neg_risk=False,
        last_trade_price="0.56",
    )


@pytest.fixture
def sample_market_metadata() -> MarketMetadata:
    """A realistic MarketMetadata from the Gamma API."""
    return MarketMetadata(**SAMPLE_MARKET_METADATA_RAW)


@pytest.fixture
def sample_event_metadata() -> EventMetadata:
    """A realistic EventMetadata from the Gamma API."""
    return EventMetadata(**SAMPLE_EVENT_METADATA_RAW)


@pytest.fixture
def polymarket_client() -> PolymarketClient:
    """A PolymarketClient with default timeout."""
    return PolymarketClient(timeout=5.0)


# ── Mock Response Helpers ────────────────────────────────────────────────


class FakeHTTPResponse:
    """Simulates urllib HTTP response for mocking."""

    def __init__(
        self,
        data: dict[str, Any],
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._data = json.dumps(data).encode("utf-8")
        self.status = status
        self.headers = headers or {"Content-Type": "application/json"}

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


def _make_fake_response(data: dict[str, Any], status: int = 200) -> FakeHTTPResponse:
    """Create a fake HTTP response for mocking."""
    return FakeHTTPResponse(data, status=status)


def _make_error_response(
    error_msg: str, status: int = 400, code: str | None = None
) -> FakeHTTPResponse:
    """Create a fake error response."""
    body: dict[str, Any] = {"error": error_msg}
    if code:
        body["code"] = code
    return FakeHTTPResponse(body, status=status)


class _FakeBytesIO:
    """File-like wrapper around bytes, matching what urllib.error.HTTPError.fp expects."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def read(self, n: int = -1) -> bytes:
        if n == -1:
            result = self._data[self._pos :]
            self._pos = len(self._data)
        else:
            result = self._data[self._pos : self._pos + n]
            self._pos += len(result)
        return result

    def close(self) -> None:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# A. RAW MODEL TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestRawModels:
    """Test that Polymarket raw Pydantic models parse correctly."""

    # ── OrderBookLevel ────────────────────────────────────────────────

    def test_order_book_level_valid(self) -> None:
        level = OrderBookLevel(price="0.55", size="100.0")
        assert level.price == "0.55"
        assert level.size == "100.0"

    def test_order_book_level_frozen(self) -> None:
        level = OrderBookLevel(price="0.55", size="100.0")
        with pytest.raises(Exception):
            level.price = "0.60"  # type: ignore[misc]

    def test_order_book_level_zero_size(self) -> None:
        level = OrderBookLevel(price="0.55", size="0.0")
        assert level.size == "0.0"

    # ── OrderBookSummary ─────────────────────────────────────────────

    def test_order_book_summary_valid(self, sample_order_book: OrderBookSummary) -> None:
        assert len(sample_order_book.bids) == 2
        assert len(sample_order_book.asks) == 2
        assert sample_order_book.market.startswith("0x")
        assert sample_order_book.neg_risk is False

    def test_order_book_summary_frozen(self, sample_order_book: OrderBookSummary) -> None:
        with pytest.raises(Exception):
            sample_order_book.hash = "changed"  # type: ignore[misc]

    def test_order_book_summary_empty_sides(self) -> None:
        book = OrderBookSummary(
            market="0xabc",
            asset_id="123",
            timestamp="1693000000000",
            hash="h1",
            bids=[],
            asks=[],
            min_order_size="1.0",
            tick_size="0.01",
            neg_risk=False,
            last_trade_price="0.0",
        )
        assert book.bids == []
        assert book.asks == []

    def test_order_book_summary_missing_required_field(self) -> None:
        with pytest.raises(Exception):
            OrderBookSummary(
                market="0xabc",
                # missing asset_id
                timestamp="1693000000000",
                hash="h1",
                bids=[],
                asks=[],
                min_order_size="1.0",
                tick_size="0.01",
                neg_risk=False,
                last_trade_price="0.0",
            )

    # ── MarketMetadata ───────────────────────────────────────────────

    def test_market_metadata_valid(self, sample_market_metadata: MarketMetadata) -> None:
        assert sample_market_metadata.id == "703257"
        assert sample_market_metadata.slug == "will-bitcoin-hit-100k-2026"
        assert sample_market_metadata.condition_id is not None
        assert sample_market_metadata.condition_id.startswith("0x")

    def test_market_metadata_frozen(self, sample_market_metadata: MarketMetadata) -> None:
        with pytest.raises(Exception):
            sample_market_metadata.id = "999"  # type: ignore[misc]

    def test_market_metadata_optional_fields(self) -> None:
        meta = MarketMetadata(id="1")
        assert meta.slug is None
        assert meta.question is None
        assert meta.condition_id is None
        assert meta.clob_token_ids is None

    def test_market_metadata_missing_id(self) -> None:
        with pytest.raises(Exception):
            MarketMetadata()  # type: ignore[call-arg]

    # ── EventMetadata ────────────────────────────────────────────────

    def test_event_metadata_valid(self, sample_event_metadata: EventMetadata) -> None:
        assert sample_event_metadata.id == "12345"
        assert sample_event_metadata.title == "2028 US Presidential Election"
        assert len(sample_event_metadata.markets) == 1

    def test_event_metadata_frozen(self, sample_event_metadata: EventMetadata) -> None:
        with pytest.raises(Exception):
            sample_event_metadata.id = "999"  # type: ignore[misc]

    def test_event_metadata_empty_markets(self) -> None:
        event = EventMetadata(id="999")
        assert event.markets == []

    # ── PriceResponse ────────────────────────────────────────────────

    def test_price_response_valid(self) -> None:
        resp = PriceResponse(price="0.56")
        assert resp.price == "0.56"

    # ── MidpointResponse ─────────────────────────────────────────────

    def test_midpoint_response_valid(self) -> None:
        resp = MidpointResponse(mid="0.565")
        assert resp.mid == "0.565"

    # ── SpreadResponse ───────────────────────────────────────────────

    def test_spread_response_valid(self) -> None:
        resp = SpreadResponse(spread="0.03")
        assert resp.spread == "0.03"

    # ── LastTradePriceResponse ───────────────────────────────────────

    def test_last_trade_price_response_valid(self) -> None:
        resp = LastTradePriceResponse(price="0.56", side="BUY")
        assert resp.price == "0.56"
        assert resp.side == "BUY"

    # ── ErrorResponse ────────────────────────────────────────────────

    def test_error_response_valid(self) -> None:
        resp = ErrorResponse(error="Rate limit exceeded", code="RATE_LIMITED")
        assert resp.error == "Rate limit exceeded"
        assert resp.code == "RATE_LIMITED"
        assert resp.retry_after_seconds is None

    def test_error_response_with_retry(self) -> None:
        resp = ErrorResponse(
            error="Rate limited", code="RATE_LIMITED", retry_after_seconds=30
        )
        assert resp.retry_after_seconds == 30

    def test_error_response_optional_fields(self) -> None:
        resp = ErrorResponse(error="Something failed")
        assert resp.code is None
        assert resp.retry_after_seconds is None


# ══════════════════════════════════════════════════════════════════════════════
# B. CLIENT TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestPolymarketClient:
    """Test client behavior with mocked HTTP responses."""

    def test_successful_get_order_book(
        self, polymarket_client: PolymarketClient
    ) -> None:
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _make_fake_response(SAMPLE_ORDER_BOOK_RAW)
            book = polymarket_client.get_order_book("12345678901234567890")
            assert isinstance(book, OrderBookSummary)
            assert len(book.bids) == 2
            assert len(book.asks) == 2
            mock_urlopen.assert_called_once()

    def test_successful_get_price(self, polymarket_client: PolymarketClient) -> None:
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _make_fake_response(SAMPLE_PRICE_RESPONSE_RAW)
            price = polymarket_client.get_price("12345678901234567890", side="BUY")
            assert isinstance(price, PriceResponse)
            assert price.price == "0.56"

    def test_successful_get_midpoint(self, polymarket_client: PolymarketClient) -> None:
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _make_fake_response(SAMPLE_MIDPOINT_RESPONSE_RAW)
            mid = polymarket_client.get_midpoint("12345678901234567890")
            assert isinstance(mid, MidpointResponse)
            assert mid.mid == "0.565"

    def test_successful_get_spread(self, polymarket_client: PolymarketClient) -> None:
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _make_fake_response(SAMPLE_SPREAD_RESPONSE_RAW)
            spread = polymarket_client.get_spread("12345678901234567890")
            assert isinstance(spread, SpreadResponse)
            assert spread.spread == "0.03"

    def test_successful_get_last_trade_price(
        self, polymarket_client: PolymarketClient
    ) -> None:
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _make_fake_response(SAMPLE_LAST_TRADE_PRICE_RAW)
            ltp = polymarket_client.get_last_trade_price("12345678901234567890")
            assert isinstance(ltp, LastTradePriceResponse)
            assert ltp.price == "0.56"
            assert ltp.side == "BUY"

    def test_successful_get_market(self, polymarket_client: PolymarketClient) -> None:
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _make_fake_response(SAMPLE_MARKET_METADATA_RAW)
            market = polymarket_client.get_market("703257")
            assert isinstance(market, MarketMetadata)
            assert market.id == "703257"

    def test_successful_get_event(self, polymarket_client: PolymarketClient) -> None:
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _make_fake_response(SAMPLE_EVENT_METADATA_RAW)
            event = polymarket_client.get_event("12345")
            assert isinstance(event, EventMetadata)
            assert event.id == "12345"
            assert len(event.markets) == 1

    def test_get_price_invalid_side_raises_value_error(
        self, polymarket_client: PolymarketClient
    ) -> None:
        with pytest.raises(ValueError, match="side must be"):
            polymarket_client.get_price("123", side="INVALID")

    def test_get_order_books_exceeds_limit_raises(self, polymarket_client: PolymarketClient) -> None:
        with pytest.raises(ValueError, match="Maximum 500"):
            polymarket_client.get_order_books([f"token_{i}" for i in range(501)])

    def test_get_order_books_success(self, polymarket_client: PolymarketClient) -> None:
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _make_fake_response(
                [SAMPLE_ORDER_BOOK_RAW, SAMPLE_ORDER_BOOK_RAW]
            )
            books = polymarket_client.get_order_books(["token1", "token2"])
            assert len(books) == 2

    # ── Error Handling ───────────────────────────────────────────────

    def test_http_error_raises_polymarket_api_error(
        self, polymarket_client: PolymarketClient
    ) -> None:
        with patch("urllib.request.urlopen") as mock_urlopen:
            error_body = json.dumps({"error": "Invalid token"}).encode("utf-8")
            http_error = urllib.error.HTTPError(
                url="https://clob.polymarket.com/book",
                code=400,
                msg="Bad Request",
                hdrs=None,
                fp=_FakeBytesIO(error_body),
            )
            mock_urlopen.side_effect = http_error
            with pytest.raises(PolymarketAPIError) as exc_info:
                polymarket_client.get_order_book("bad_token")
            assert exc_info.value.status_code == 400
            assert "Invalid token" in str(exc_info.value)

    def test_http_error_with_malformed_body_raises_api_error(
        self, polymarket_client: PolymarketClient
    ) -> None:
        with patch("urllib.request.urlopen") as mock_urlopen:
            http_error = urllib.error.HTTPError(
                url="https://clob.polymarket.com/book",
                code=500,
                msg="Internal Server Error",
                hdrs=None,
                fp=_FakeBytesIO(b"not json at all"),
            )
            mock_urlopen.side_effect = http_error
            with pytest.raises(PolymarketAPIError) as exc_info:
                polymarket_client.get_order_book("token")
            assert exc_info.value.status_code == 500

    def test_timeout_raises_polymarket_timeout_error(
        self, polymarket_client: PolymarketClient
    ) -> None:
        with patch("urllib.request.urlopen") as mock_urlopen:
            url_error = urllib.error.URLError(reason="timed out")
            mock_urlopen.side_effect = url_error
            with pytest.raises(PolymarketTimeoutError, match="timed out"):
                polymarket_client.get_order_book("token")

    def test_connection_failure_raises_polymarket_connection_error(
        self, polymarket_client: PolymarketClient
    ) -> None:
        with patch("urllib.request.urlopen") as mock_urlopen:
            url_error = urllib.error.URLError(reason="Connection refused")
            mock_urlopen.side_effect = url_error
            with pytest.raises(PolymarketConnectionError, match="Connection refused"):
                polymarket_client.get_order_book("token")

    def test_malformed_json_raises_client_error(
        self, polymarket_client: PolymarketClient
    ) -> None:
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = b"{invalid json"
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            with pytest.raises(PolymarketClientError, match="Invalid JSON"):
                polymarket_client.get_order_book("token")

    def test_invalid_response_schema_raises_error(
        self, polymarket_client: PolymarketClient
    ) -> None:
        with patch("urllib.request.urlopen") as mock_urlopen:
            # Return a dict that doesn't match OrderBookSummary schema
            mock_urlopen.return_value = _make_fake_response(
                {"completely": "wrong", "fields": True}
            )
            with pytest.raises(Exception):  # Pydantic validation error
                polymarket_client.get_order_book("token")

    def test_error_hierarchy(self) -> None:
        """PolymarketClientError inherits from MarketDataError."""
        assert issubclass(PolymarketClientError, PolymarketClientError)
        assert issubclass(PolymarketAPIError, PolymarketClientError)
        assert issubclass(PolymarketConnectionError, PolymarketClientError)
        assert issubclass(PolymarketTimeoutError, PolymarketClientError)


# ══════════════════════════════════════════════════════════════════════════════
# C. NORMALIZER TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestPolymarketNormalizer:
    """Test normalization logic."""

    def _make_normalizer(self):
        """Create a PolymarketNormalizer with a deterministic clock."""
        from quant_engine.providers.polymarket.normalizer import PolymarketNormalizer

        fixed_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        return PolymarketNormalizer(clock=lambda: fixed_time)

    def test_best_bid_is_max_of_all_bid_levels(self) -> None:
        """Best bid should be the HIGHEST bid price."""
        normalizer = self._make_normalizer()
        book = OrderBookSummary(
            market="0xabc",
            asset_id="123",
            timestamp="1693000000000",
            hash="h1",
            bids=[
                OrderBookLevel(price="0.50", size="100"),
                OrderBookLevel(price="0.55", size="200"),
                OrderBookLevel(price="0.45", size="50"),
            ],
            asks=[],
            min_order_size="1",
            tick_size="0.01",
            neg_risk=False,
            last_trade_price="0.52",
        )
        quote = normalizer.normalize_order_book(
            token_id=book.asset_id, order_book=book
        )
        assert quote.bid_price == 0.55

    def test_best_ask_is_min_of_all_ask_levels(self) -> None:
        """Best ask should be the LOWEST ask price."""
        normalizer = self._make_normalizer()
        book = OrderBookSummary(
            market="0xabc",
            asset_id="123",
            timestamp="1693000000000",
            hash="h1",
            bids=[],
            asks=[
                OrderBookLevel(price="0.60", size="100"),
                OrderBookLevel(price="0.58", size="150"),
                OrderBookLevel(price="0.65", size="50"),
            ],
            min_order_size="1",
            tick_size="0.01",
            neg_risk=False,
            last_trade_price="0.59",
        )
        quote = normalizer.normalize_order_book(
            token_id=book.asset_id, order_book=book
        )
        assert quote.ask_price == 0.58

    def test_provider_identifier_is_polymarket(self) -> None:
        normalizer = self._make_normalizer()
        book = OrderBookSummary(
            market="0xabc",
            asset_id="123",
            timestamp="1693000000000",
            hash="h1",
            bids=[],
            asks=[],
            min_order_size="1",
            tick_size="0.01",
            neg_risk=False,
            last_trade_price="0.0",
        )
        quote = normalizer.normalize_order_book(
            token_id=book.asset_id, order_book=book
        )
        assert quote.provider == "polymarket"

    def test_provider_instrument_id_matches_asset_id(self) -> None:
        normalizer = self._make_normalizer()
        book = OrderBookSummary(
            market="0xabc",
            asset_id="9876543210",
            timestamp="1693000000000",
            hash="h1",
            bids=[],
            asks=[],
            min_order_size="1",
            tick_size="0.01",
            neg_risk=False,
            last_trade_price="0.0",
        )
        quote = normalizer.normalize_order_book(
            token_id=book.asset_id, order_book=book
        )
        assert quote.provider_instrument_id == "9876543210"

    def test_source_timestamp_correctly_parsed(self) -> None:
        """Epoch milliseconds '1693000000000' → 2023-08-26T14:40:00+00:00."""
        normalizer = self._make_normalizer()
        book = OrderBookSummary(
            market="0xabc",
            asset_id="123",
            timestamp="1693000000000",
            hash="h1",
            bids=[],
            asks=[],
            min_order_size="1",
            tick_size="0.01",
            neg_risk=False,
            last_trade_price="0.0",
        )
        quote = normalizer.normalize_order_book(
            token_id=book.asset_id, order_book=book
        )
        assert quote.source_timestamp.tzinfo is not None
        assert quote.source_timestamp.year == 2023

    def test_ingestion_timestamp_is_set(self) -> None:
        normalizer = self._make_normalizer()
        book = OrderBookSummary(
            market="0xabc",
            asset_id="123",
            timestamp="1693000000000",
            hash="h1",
            bids=[],
            asks=[],
            min_order_size="1",
            tick_size="0.01",
            neg_risk=False,
            last_trade_price="0.0",
        )
        quote = normalizer.normalize_order_book(
            token_id=book.asset_id, order_book=book
        )
        assert quote.ingestion_timestamp.tzinfo is not None
        # Fixed clock returns deterministic ingestion timestamp
        assert quote.ingestion_timestamp.year == 2025

    def test_deterministic_output(self) -> None:
        """Same input → same record_id."""
        normalizer = self._make_normalizer()
        book = OrderBookSummary(
            market="0xabc",
            asset_id="123",
            timestamp="1693000000000",
            hash="h1",
            bids=[OrderBookLevel(price="0.5", size="10")],
            asks=[],
            min_order_size="1",
            tick_size="0.01",
            neg_risk=False,
            last_trade_price="0.5",
        )
        q1 = normalizer.normalize_order_book(
            token_id=book.asset_id, order_book=book
        )
        q2 = normalizer.normalize_order_book(
            token_id=book.asset_id, order_book=book
        )
        assert q1.record_id == q2.record_id

    def test_empty_book_returns_none_bid_ask(self) -> None:
        normalizer = self._make_normalizer()
        book = OrderBookSummary(
            market="0xabc",
            asset_id="123",
            timestamp="1693000000000",
            hash="h1",
            bids=[],
            asks=[],
            min_order_size="1",
            tick_size="0.01",
            neg_risk=False,
            last_trade_price="0.0",
        )
        quote = normalizer.normalize_order_book(
            token_id=book.asset_id, order_book=book
        )
        assert quote.bid_price is None
        assert quote.ask_price is None

    def test_one_sided_book_bids_only(self) -> None:
        normalizer = self._make_normalizer()
        book = OrderBookSummary(
            market="0xabc",
            asset_id="123",
            timestamp="1693000000000",
            hash="h1",
            bids=[OrderBookLevel(price="0.50", size="100")],
            asks=[],
            min_order_size="1",
            tick_size="0.01",
            neg_risk=False,
            last_trade_price="0.50",
        )
        quote = normalizer.normalize_order_book(
            token_id=book.asset_id, order_book=book
        )
        assert quote.bid_price == 0.50
        assert quote.ask_price is None

    def test_one_sided_book_asks_only(self) -> None:
        normalizer = self._make_normalizer()
        book = OrderBookSummary(
            market="0xabc",
            asset_id="123",
            timestamp="1693000000000",
            hash="h1",
            bids=[],
            asks=[OrderBookLevel(price="0.60", size="100")],
            min_order_size="1",
            tick_size="0.01",
            neg_risk=False,
            last_trade_price="0.60",
        )
        quote = normalizer.normalize_order_book(
            token_id=book.asset_id, order_book=book
        )
        assert quote.bid_price is None
        assert quote.ask_price == 0.60

    def test_bid_size_matches_best_bid_level(self) -> None:
        normalizer = self._make_normalizer()
        book = OrderBookSummary(
            market="0xabc",
            asset_id="123",
            timestamp="1693000000000",
            hash="h1",
            bids=[
                OrderBookLevel(price="0.50", size="100"),
                OrderBookLevel(price="0.55", size="200"),
            ],
            asks=[],
            min_order_size="1",
            tick_size="0.01",
            neg_risk=False,
            last_trade_price="0.52",
        )
        quote = normalizer.normalize_order_book(
            token_id=book.asset_id, order_book=book
        )
        assert quote.bid_size == 200.0

    def test_ask_size_matches_best_ask_level(self) -> None:
        normalizer = self._make_normalizer()
        book = OrderBookSummary(
            market="0xabc",
            asset_id="123",
            timestamp="1693000000000",
            hash="h1",
            bids=[],
            asks=[
                OrderBookLevel(price="0.58", size="150"),
                OrderBookLevel(price="0.60", size="100"),
            ],
            min_order_size="1",
            tick_size="0.01",
            neg_risk=False,
            last_trade_price="0.59",
        )
        quote = normalizer.normalize_order_book(
            token_id=book.asset_id, order_book=book
        )
        assert quote.ask_size == 150.0

    def test_empty_token_id_raises_normalization_error(self) -> None:
        """Empty token_id must be rejected."""
        from quant_engine.market_data.errors import NormalizationError

        normalizer = self._make_normalizer()
        book = OrderBookSummary(
            market="0xabc",
            asset_id="123",
            timestamp="1693000000000",
            hash="h1",
            bids=[],
            asks=[],
            min_order_size="1",
            tick_size="0.01",
            neg_risk=False,
            last_trade_price="0.0",
        )
        with pytest.raises(NormalizationError, match="token_id must be non-empty"):
            normalizer.normalize_order_book(token_id="", order_book=book)

    def test_whitespace_token_id_raises_normalization_error(self) -> None:
        """Whitespace-only token_id must be rejected."""
        from quant_engine.market_data.errors import NormalizationError

        normalizer = self._make_normalizer()
        book = OrderBookSummary(
            market="0xabc",
            asset_id="123",
            timestamp="1693000000000",
            hash="h1",
            bids=[],
            asks=[],
            min_order_size="1",
            tick_size="0.01",
            neg_risk=False,
            last_trade_price="0.0",
        )
        with pytest.raises(NormalizationError, match="token_id must be non-empty"):
            normalizer.normalize_order_book(token_id="   ", order_book=book)

    def test_multiple_levels_same_price_sums_size(self) -> None:
        """When multiple bid levels share the same price, sizes are summed."""
        normalizer = self._make_normalizer()
        book = OrderBookSummary(
            market="0xabc",
            asset_id="123",
            timestamp="1693000000000",
            hash="h1",
            bids=[
                OrderBookLevel(price="0.55", size="100"),
                OrderBookLevel(price="0.55", size="50"),
                OrderBookLevel(price="0.50", size="200"),
            ],
            asks=[],
            min_order_size="1",
            tick_size="0.01",
            neg_risk=False,
            last_trade_price="0.55",
        )
        quote = normalizer.normalize_order_book(
            token_id=book.asset_id, order_book=book
        )
        assert quote.bid_price == 0.55
        assert quote.bid_size == 150.0

    def test_normalize_from_raw_dict(self) -> None:
        """normalize() accepts a raw dict and produces a MarketQuote."""
        normalizer = self._make_normalizer()
        quote = normalizer.normalize(SAMPLE_ORDER_BOOK_RAW)
        assert quote.provider == "polymarket"
        assert quote.provider_instrument_id == "12345678901234567890"

    def test_normalize_rejects_invalid_raw_dict(self) -> None:
        """normalize() raises NormalizationError for malformed dicts."""
        from quant_engine.market_data.errors import NormalizationError

        normalizer = self._make_normalizer()
        with pytest.raises(NormalizationError, match="Failed to parse"):
            normalizer.normalize({"completely": "wrong"})

    def test_normalize_rejects_invalid_timestamp(self) -> None:
        """normalize() raises NormalizationError for unparseable timestamps."""
        from quant_engine.market_data.errors import NormalizationError

        normalizer = self._make_normalizer()
        raw = dict(SAMPLE_ORDER_BOOK_RAW)
        raw["timestamp"] = "not-a-number"
        with pytest.raises(NormalizationError, match="Invalid epoch timestamp"):
            normalizer.normalize(raw)


# ══════════════════════════════════════════════════════════════════════════════
# D. INTEGRATION TESTS (end-to-end with fixtures)
# ══════════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """End-to-end flow tests: raw dict → normalizer → MarketQuote → validate_quote."""

    def _make_normalizer(self):
        """Create a PolymarketNormalizer with a deterministic clock."""
        from quant_engine.providers.polymarket.normalizer import PolymarketNormalizer

        fixed_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        return PolymarketNormalizer(clock=lambda: fixed_time)

    def test_full_flow_raw_dict_to_valid_quote(self) -> None:
        """Raw dict → normalize() → MarketQuote → validate_quote passes."""
        normalizer = self._make_normalizer()
        quote = normalizer.normalize(SAMPLE_ORDER_BOOK_RAW)
        errors = validate_quote(quote)
        assert errors == [], f"Validation failed: {errors}"
        assert quote.provider == "polymarket"
        assert quote.bid_price == 0.55
        assert quote.ask_price == 0.58

    def test_full_flow_empty_book_to_valid_quote(self) -> None:
        """Empty book → quote with None sides → still passes validation."""
        normalizer = self._make_normalizer()
        raw = dict(SAMPLE_ORDER_BOOK_RAW)
        raw["bids"] = []
        raw["asks"] = []
        quote = normalizer.normalize(raw)
        errors = validate_quote(quote)
        assert errors == []
        assert quote.bid_price is None
        assert quote.ask_price is None

    def test_full_flow_dataset_deterministic(self) -> None:
        """Same raw input → same Dataset dataset_id."""
        normalizer = self._make_normalizer()
        q1 = normalizer.normalize(SAMPLE_ORDER_BOOK_RAW)
        q2 = normalizer.normalize(SAMPLE_ORDER_BOOK_RAW)
        ds = Dataset(records=[q1, q2])
        # Same record_ids → deterministic dataset_id
        assert ds.dataset_id is not None

    def test_full_flow_multiple_books_to_dataset(self) -> None:
        """Multiple order books → Dataset with correct record count."""
        normalizer = self._make_normalizer()
        raw2 = dict(SAMPLE_ORDER_BOOK_RAW)
        raw2["asset_id"] = "99999"
        q1 = normalizer.normalize(SAMPLE_ORDER_BOOK_RAW)
        q2 = normalizer.normalize(raw2)
        ds = Dataset(records=[q1, q2])
        assert len(ds.records) == 2
        assert q1.provider_instrument_id != q2.provider_instrument_id

    def test_full_flow_client_to_normalizer_to_validation(self) -> None:
        """Client response dict → OrderBookSummary → normalize_order_book → validate."""
        normalizer = self._make_normalizer()
        book = OrderBookSummary(**SAMPLE_ORDER_BOOK_RAW)
        quote = normalizer.normalize_order_book(
            token_id=book.asset_id, order_book=book
        )
        errors = validate_quote(quote)
        assert errors == [], f"Validation failed: {errors}"

    def test_full_flow_dataset_to_dict_roundtrip(self) -> None:
        """Dataset → to_dict → from_dict preserves data."""
        normalizer = self._make_normalizer()
        q1 = normalizer.normalize(SAMPLE_ORDER_BOOK_RAW)
        ds = Dataset(records=[q1])
        ds_dict = ds.to_dict()
        ds2 = Dataset.from_dict(ds_dict)
        assert ds2.dataset_id == ds.dataset_id
        assert len(ds2.records) == 1


# ══════════════════════════════════════════════════════════════════════════════
# E. EDGE CASE TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge case tests for models and validation."""

    def test_single_level_each_side(self) -> None:
        """Single level on each side — best bid/ask extraction still works."""
        book = OrderBookSummary(
            market="0xabc",
            asset_id="123",
            timestamp="1693000000000",
            hash="h1",
            bids=[OrderBookLevel(price="0.50", size="100")],
            asks=[OrderBookLevel(price="0.60", size="200")],
            min_order_size="1",
            tick_size="0.01",
            neg_risk=False,
            last_trade_price="0.55",
        )
        assert len(book.bids) == 1
        assert len(book.asks) == 1

    def test_many_levels_best_bid_ask_extraction(self) -> None:
        """100 levels — best bid is max, best ask is min."""
        bids = [OrderBookLevel(price=f"0.{i:02d}", size="10") for i in range(50)]
        asks = [OrderBookLevel(price=f"0.{i + 50:02d}", size="10") for i in range(50)]
        book = OrderBookSummary(
            market="0xabc",
            asset_id="123",
            timestamp="1693000000000",
            hash="h1",
            bids=bids,
            asks=asks,
            min_order_size="1",
            tick_size="0.01",
            neg_risk=False,
            last_trade_price="0.50",
        )
        # Best bid = max price = "0.49", best ask = min price = "0.50"
        assert book.bids[-1].price == "0.49"
        assert book.asks[0].price == "0.50"

    def test_zero_size_level(self) -> None:
        """Zero-size levels are valid in the model."""
        level = OrderBookLevel(price="0.50", size="0.0")
        assert level.size == "0.0"

    def test_negative_price_in_model(self) -> None:
        """Model accepts negative price — validation should catch it."""
        # The model is a raw schema; validation is downstream
        level = OrderBookLevel(price="-0.10", size="100")
        assert level.price == "-0.10"

    def test_validate_rejects_negative_price(self) -> None:
        """validate_quote rejects negative bid/ask prices."""
        quote = MarketQuote(
            source_timestamp=_utc("2026-09-04T01:00:00+00:00"),
            ingestion_timestamp=_utc("2026-09-04T01:00:01+00:00"),
            provider="polymarket",
            provider_instrument_id="123",
            bid_price=-0.10,
            ask_price=0.60,
        )
        errors = validate_quote(quote)
        assert any("negative" in e for e in errors)

    def test_validate_rejects_nan_price(self) -> None:
        """validate_quote rejects NaN in price fields."""
        quote = MarketQuote(
            source_timestamp=_utc("2026-09-04T01:00:00+00:00"),
            ingestion_timestamp=_utc("2026-09-04T01:00:01+00:00"),
            provider="polymarket",
            provider_instrument_id="123",
            bid_price=float("nan"),
        )
        errors = validate_quote(quote)
        assert any("NaN" in e for e in errors)

    def test_validate_rejects_inf_price(self) -> None:
        """validate_quote rejects infinity in price fields."""
        quote = MarketQuote(
            source_timestamp=_utc("2026-09-04T01:00:00+00:00"),
            ingestion_timestamp=_utc("2026-09-04T01:00:01+00:00"),
            provider="polymarket",
            provider_instrument_id="123",
            ask_price=float("inf"),
        )
        errors = validate_quote(quote)
        assert any("infinity" in e for e in errors)

    def test_validate_rejects_crossed_market(self) -> None:
        """validate_quote rejects bid > ask."""
        quote = MarketQuote(
            source_timestamp=_utc("2026-09-04T01:00:00+00:00"),
            ingestion_timestamp=_utc("2026-09-04T01:00:01+00:00"),
            provider="polymarket",
            provider_instrument_id="123",
            bid_price=0.70,
            ask_price=0.50,
        )
        errors = validate_quote(quote)
        assert any("crossed" in e for e in errors)

    def test_validate_accepts_boundary_prices(self) -> None:
        """Prices at exactly 0.0 and 1.0 are valid for binary markets."""
        quote = MarketQuote(
            source_timestamp=_utc("2026-09-04T01:00:00+00:00"),
            ingestion_timestamp=_utc("2026-09-04T01:00:01+00:00"),
            provider="polymarket",
            provider_instrument_id="123",
            bid_price=0.0,
            ask_price=1.0,
        )
        errors = validate_quote(quote)
        assert errors == []

    def test_order_book_summary_preserves_all_fields(self) -> None:
        """All fields from raw response are preserved in the model."""
        book = OrderBookSummary(**SAMPLE_ORDER_BOOK_RAW)
        assert book.min_order_size == "1.0"
        assert book.tick_size == "0.01"
        assert book.neg_risk is False
        assert book.last_trade_price == "0.56"
        assert book.hash == "a1b2c3d4e5f6"
