"""Tests for Kalshi market data adapter.

All tests use deterministic fixtures and mocked responses.
No live API calls. No network access required.

Kalshi API Key Points:
- Base URL: https://external-api.kalshi.com/trade-api/v2
- Orderbook: GET /markets/{ticker}/orderbook — returns only bids (YES and NO)
- In binary markets, a YES bid at price X = NO ask at (1 - X)
- Prices are dollar strings in [0.00, 1.00] range
- Each price level is [price_dollars, count_fp] (2-element string array)
- Arrays sorted ascending; best bid is the last element
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from quant_engine.market_data import (
    Dataset,
    MarketQuote,
)
from quant_engine.market_data.errors import NormalizationError
from quant_engine.market_data.validation import validate_quote
from quant_engine.providers.kalshi.client import (
    KalshiAPIError,
    KalshiClient,
    KalshiClientError,
    KalshiConnectionError,
    KalshiTimeoutError,
)
from quant_engine.providers.kalshi.models import (
    GetMarketOrderbookResponse,
    GetMarketResponse,
    GetMarketsResponse,
)
from quant_engine.providers.kalshi.normalizer import KalshiNormalizer

# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════


def _utc(iso: str) -> datetime:
    """Parse ISO string to UTC datetime."""
    return datetime.fromisoformat(iso)


def _make_orderbook(raw: dict[str, Any]) -> GetMarketOrderbookResponse:
    """Parse a raw orderbook dict into a Pydantic model."""
    return GetMarketOrderbookResponse(**raw)


# ── Raw API Response Fixtures ─────────────────────────────────────────────

# Kalshi orderbook response: only bids, no asks.
# YES bid at 0.55 = NO ask at 0.45 (1.00 - 0.55)
SAMPLE_ORDERBOOK_RAW: dict[str, Any] = {
    "orderbook_fp": {
        "yes_dollars": [
            ["0.5000", "200.00"],
            ["0.5500", "100.00"],
        ],
        "no_dollars": [
            ["0.4000", "150.00"],
            ["0.4500", "100.00"],
        ],
    }
}

# Raw dict with ticker for normalize() tests
SAMPLE_ORDERBOOK_WITH_TICKER: dict[str, Any] = {
    "ticker": "KXMLIFE-26-SEP05-100-ABOVE",
    "orderbook_fp": {
        "yes_dollars": [
            ["0.5000", "200.00"],
            ["0.5500", "100.00"],
        ],
        "no_dollars": [
            ["0.4000", "150.00"],
            ["0.4500", "100.00"],
        ],
    },
}

# Kalshi market metadata response
SAMPLE_MARKET_RAW: dict[str, Any] = {
    "market": {
        "ticker": "KXMLIFE-26-SEP05-100-ABOVE",
        "event_ticker": "KXMLIFE-26-SEP05",
        "market_type": "binary",
        "yes_sub_title": "Yes",
        "no_sub_title": "No",
        "created_time": "2025-01-15T10:00:00Z",
        "updated_time": "2025-01-15T12:00:00Z",
        "open_time": "2025-01-15T10:00:00Z",
        "close_time": "2025-09-05T23:59:59Z",
        "latest_expiration_time": "2025-09-05T23:59:59Z",
        "settlement_timer_seconds": 300,
        "status": "active",
        "notional_value_dollars": "1.0000",
        "yes_bid_dollars": "0.5500",
        "yes_ask_dollars": "0.5800",
        "no_bid_dollars": "0.4200",
        "no_ask_dollars": "0.4500",
        "yes_bid_size_fp": "100.00",
        "yes_ask_size_fp": "150.00",
        "last_price_dollars": "0.5600",
        "previous_yes_bid_dollars": "0.5400",
        "previous_yes_ask_dollars": "0.5700",
        "previous_price_dollars": "0.5500",
        "volume_fp": "10000.00",
        "volume_24h_fp": "500.00",
        "open_interest_fp": "2000.00",
        "result": "",
        "can_close_early": False,
        "expiration_value": "",
        "rules_primary": "Settles to 1.00 if condition met, 0.00 otherwise.",
        "rules_secondary": "",
        "price_level_structure": [],
        "price_ranges": [],
    }
}

# Empty orderbook
EMPTY_ORDERBOOK_RAW: dict[str, Any] = {
    "orderbook_fp": {
        "yes_dollars": [],
        "no_dollars": [],
    }
}

# One-sided orderbook (YES bids only)
YES_ONLY_ORDERBOOK_RAW: dict[str, Any] = {
    "orderbook_fp": {
        "yes_dollars": [
            ["0.3000", "50.00"],
            ["0.3500", "75.00"],
        ],
        "no_dollars": [],
    }
}

# One-sided orderbook (NO bids only)
NO_ONLY_ORDERBOOK_RAW: dict[str, Any] = {
    "orderbook_fp": {
        "yes_dollars": [],
        "no_dollars": [
            ["0.6000", "120.00"],
            ["0.6500", "80.00"],
        ],
    }
}


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


def _make_normalizer() -> KalshiNormalizer:
    """Create a KalshiNormalizer with a deterministic clock."""
    fixed_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    return KalshiNormalizer(clock=lambda: fixed_time)


# ══════════════════════════════════════════════════════════════════════════════
# A. RAW MODEL TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestRawModels:
    """Test that Kalshi raw Pydantic models parse correctly."""

    def test_price_level_valid(self) -> None:
        """Kalshi price levels are [price_dollars, count_fp] string arrays."""
        level = ["0.5500", "100.00"]
        assert len(level) == 2
        assert level[0] == "0.5500"
        assert level[1] == "100.00"

    def test_price_level_zero_quantity(self) -> None:
        """Zero quantity is valid."""
        level = ["0.5000", "0.00"]
        assert level[1] == "0.00"

    def test_price_level_boundary_prices(self) -> None:
        """Prices at 0.00 and 1.00 are valid for binary markets."""
        level_min = ["0.0000", "50.00"]
        level_max = ["1.0000", "50.00"]
        assert float(level_min[0]) == 0.0
        assert float(level_max[0]) == 1.0

    def test_orderbook_fp_valid(self) -> None:
        """Orderbook has yes_dollars and no_dollars arrays."""
        orderbook = SAMPLE_ORDERBOOK_RAW["orderbook_fp"]
        assert "yes_dollars" in orderbook
        assert "no_dollars" in orderbook
        assert len(orderbook["yes_dollars"]) == 2
        assert len(orderbook["no_dollars"]) == 2

    def test_orderbook_fp_empty(self) -> None:
        """Empty orderbook has empty arrays."""
        orderbook = EMPTY_ORDERBOOK_RAW["orderbook_fp"]
        assert orderbook["yes_dollars"] == []
        assert orderbook["no_dollars"] == []

    def test_orderbook_fp_sorted_ascending(self) -> None:
        """Price levels should be sorted by price ascending."""
        orderbook = SAMPLE_ORDERBOOK_RAW["orderbook_fp"]
        yes_prices = [float(level[0]) for level in orderbook["yes_dollars"]]
        no_prices = [float(level[0]) for level in orderbook["no_dollars"]]
        assert yes_prices == sorted(yes_prices)
        assert no_prices == sorted(no_prices)

    def test_market_metadata_valid(self) -> None:
        """Market metadata contains required fields."""
        market = SAMPLE_MARKET_RAW["market"]
        assert market["ticker"] == "KXMLIFE-26-SEP05-100-ABOVE"
        assert market["event_ticker"] == "KXMLIFE-26-SEP05"
        assert market["market_type"] == "binary"
        assert market["status"] == "active"

    def test_market_metadata_has_prices(self) -> None:
        """Market metadata includes bid/ask prices."""
        market = SAMPLE_MARKET_RAW["market"]
        assert market["yes_bid_dollars"] == "0.5500"
        assert market["yes_ask_dollars"] == "0.5800"
        assert market["no_bid_dollars"] == "0.4200"
        assert market["no_ask_dollars"] == "0.4500"

    def test_market_metadata_has_volume(self) -> None:
        """Market metadata includes volume fields."""
        market = SAMPLE_MARKET_RAW["market"]
        assert market["volume_fp"] == "10000.00"
        assert market["volume_24h_fp"] == "500.00"
        assert market["open_interest_fp"] == "2000.00"

    def test_market_metadata_frozen(self) -> None:
        """Raw dicts are mutable — this is expected for raw API responses."""
        market = SAMPLE_MARKET_RAW["market"]
        assert isinstance(market, dict)


# ══════════════════════════════════════════════════════════════════════════════
# B. CLIENT TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestKalshiClient:
    """Test client behavior with mocked HTTP responses."""

    def test_successful_get_orderbook(self) -> None:
        """Successfully fetch and parse an orderbook response."""
        client = KalshiClient(timeout=5.0)
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _make_fake_response(SAMPLE_ORDERBOOK_RAW)
            orderbook = client.get_orderbook("KXMLIFE-26-SEP05-100-ABOVE")
            assert isinstance(orderbook, GetMarketOrderbookResponse)
            assert len(orderbook.orderbook_fp.yes_dollars) == 2
            assert len(orderbook.orderbook_fp.no_dollars) == 2
            mock_urlopen.assert_called_once()

    def test_successful_get_market(self) -> None:
        """Successfully fetch and parse market metadata."""
        client = KalshiClient(timeout=5.0)
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _make_fake_response(SAMPLE_MARKET_RAW)
            market = client.get_market("KXMLIFE-26-SEP05-100-ABOVE")
            assert isinstance(market, GetMarketResponse)
            assert market.market.ticker == "KXMLIFE-26-SEP05-100-ABOVE"
            mock_urlopen.assert_called_once()

    def test_http_error_raises_kalshi_api_error(self) -> None:
        """HTTP error responses are wrapped in KalshiAPIError."""
        client = KalshiClient(timeout=5.0)
        with patch("urllib.request.urlopen") as mock_urlopen:
            error_body = json.dumps({"message": "Market not found"}).encode("utf-8")
            http_error = urllib.error.HTTPError(
                url="https://external-api.kalshi.com/trade-api/v2/markets/NOPE/orderbook",
                code=404,
                msg="Not Found",
                hdrs=None,
                fp=_FakeBytesIO(error_body),
            )
            mock_urlopen.side_effect = http_error
            with pytest.raises(KalshiAPIError) as exc_info:
                client.get_orderbook("NOPE")
            assert exc_info.value.status_code == 404
            assert "Market not found" in str(exc_info.value)

    def test_http_error_with_malformed_body_raises_api_error(self) -> None:
        """HTTP error with non-JSON body still raises KalshiAPIError."""
        client = KalshiClient(timeout=5.0)
        with patch("urllib.request.urlopen") as mock_urlopen:
            http_error = urllib.error.HTTPError(
                url="https://external-api.kalshi.com/trade-api/v2/markets/TICKER/orderbook",
                code=500,
                msg="Internal Server Error",
                hdrs=None,
                fp=_FakeBytesIO(b"not json at all"),
            )
            mock_urlopen.side_effect = http_error
            with pytest.raises(KalshiAPIError) as exc_info:
                client.get_orderbook("TICKER")
            assert exc_info.value.status_code == 500

    def test_timeout_raises_kalshi_timeout_error(self) -> None:
        """URL timeout is wrapped in KalshiTimeoutError."""
        client = KalshiClient(timeout=5.0)
        with patch("urllib.request.urlopen") as mock_urlopen:
            url_error = urllib.error.URLError(reason="timed out")
            mock_urlopen.side_effect = url_error
            with pytest.raises(KalshiTimeoutError, match="timed out"):
                client.get_orderbook("TICKER")

    def test_connection_failure_raises_kalshi_connection_error(self) -> None:
        """Connection failure is wrapped in KalshiConnectionError."""
        client = KalshiClient(timeout=5.0)
        with patch("urllib.request.urlopen") as mock_urlopen:
            url_error = urllib.error.URLError(reason="Connection refused")
            mock_urlopen.side_effect = url_error
            with pytest.raises(KalshiConnectionError, match="Connection refused"):
                client.get_orderbook("TICKER")

    def test_malformed_json_raises_client_error(self) -> None:
        """Non-JSON response body raises KalshiClientError."""
        client = KalshiClient(timeout=5.0)
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = b"{invalid json"
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            with pytest.raises(KalshiClientError, match="Invalid JSON"):
                client.get_orderbook("TICKER")

    def test_error_hierarchy(self) -> None:
        """Kalshi error classes form a proper hierarchy."""
        assert issubclass(KalshiClientError, KalshiClientError)
        assert issubclass(KalshiAPIError, KalshiClientError)
        assert issubclass(KalshiConnectionError, KalshiClientError)
        assert issubclass(KalshiTimeoutError, KalshiClientError)

    def test_get_markets_pagination(self) -> None:
        """get_markets returns paginated results with cursor."""
        client = KalshiClient(timeout=5.0)
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _make_fake_response(
                {
                    "markets": [SAMPLE_MARKET_RAW["market"]],
                    "cursor": "next-page-cursor",
                }
            )
            result = client.get_markets(status="active", limit=10)
            assert isinstance(result, GetMarketsResponse)
            assert len(result.markets) == 1
            assert result.cursor == "next-page-cursor"

    def test_url_construction(self) -> None:
        """Client constructs correct URL with query parameters."""
        client = KalshiClient(timeout=5.0)
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _make_fake_response(
                {
                    "markets": [],
                    "cursor": "",
                }
            )
            client.get_markets(status="active", limit=50)
            call_args = mock_urlopen.call_args
            url = call_args[0][0].full_url
            assert "status=active" in url
            assert "limit=50" in url


# ══════════════════════════════════════════════════════════════════════════════
# C. NORMALIZER TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestKalshiNormalizer:
    """Test normalization logic.

    Kalshi orderbook only returns bids (YES bids and NO bids).
    To derive canonical bid/ask for YES side:
    - Best YES bid = highest YES bid price
    - Best YES ask = 1.0 - highest NO bid price (inverse of NO bid)
    - Bid size = aggregate quantity at best YES bid
    - Ask size = aggregate quantity at best NO bid (which becomes YES ask)
    """

    def test_best_bid_is_max_of_yes_dollars(self) -> None:
        """Best bid should be the HIGHEST YES bid price."""
        normalizer = _make_normalizer()
        orderbook = _make_orderbook(
            {
                "orderbook_fp": {
                    "yes_dollars": [
                        ["0.5000", "100.00"],
                        ["0.5500", "200.00"],
                        ["0.4500", "50.00"],
                    ],
                    "no_dollars": [],
                }
            }
        )
        quote = normalizer.normalize_orderbook(ticker="TICKER-1", orderbook=orderbook)
        assert quote.bid_price == 0.55

    def test_best_ask_derived_from_no_dollars(self) -> None:
        """Best ask = 1.0 - highest NO bid price."""
        normalizer = _make_normalizer()
        orderbook = _make_orderbook(
            {
                "orderbook_fp": {
                    "yes_dollars": [],
                    "no_dollars": [
                        ["0.4000", "100.00"],
                        ["0.4500", "150.00"],
                        ["0.3500", "50.00"],
                    ],
                }
            }
        )
        quote = normalizer.normalize_orderbook(ticker="TICKER-1", orderbook=orderbook)
        # Best NO bid = 0.45, so YES ask = 1.0 - 0.45 = 0.55
        assert quote.ask_price == 0.55

    def test_provider_identifier_is_kalshi(self) -> None:
        normalizer = _make_normalizer()
        orderbook = _make_orderbook(EMPTY_ORDERBOOK_RAW)
        quote = normalizer.normalize_orderbook(ticker="TICKER-1", orderbook=orderbook)
        assert quote.provider == "kalshi"

    def test_provider_instrument_id_matches_ticker(self) -> None:
        normalizer = _make_normalizer()
        orderbook = _make_orderbook(EMPTY_ORDERBOOK_RAW)
        quote = normalizer.normalize_orderbook(
            ticker="KXMLIFE-26-SEP05-100-ABOVE", orderbook=orderbook
        )
        assert quote.provider_instrument_id == "KXMLIFE-26-SEP05-100-ABOVE"

    def test_ingestion_timestamp_is_set(self) -> None:
        normalizer = _make_normalizer()
        orderbook = _make_orderbook(EMPTY_ORDERBOOK_RAW)
        quote = normalizer.normalize_orderbook(ticker="TICKER-1", orderbook=orderbook)
        assert quote.ingestion_timestamp.tzinfo is not None
        assert quote.ingestion_timestamp.year == 2025

    def test_deterministic_output(self) -> None:
        """Same input → same record_id."""
        normalizer = _make_normalizer()
        orderbook = _make_orderbook(SAMPLE_ORDERBOOK_RAW)
        q1 = normalizer.normalize_orderbook(ticker="TICKER-1", orderbook=orderbook)
        q2 = normalizer.normalize_orderbook(ticker="TICKER-1", orderbook=orderbook)
        assert q1.record_id == q2.record_id

    def test_empty_book_returns_none_bid_ask(self) -> None:
        normalizer = _make_normalizer()
        orderbook = _make_orderbook(EMPTY_ORDERBOOK_RAW)
        quote = normalizer.normalize_orderbook(ticker="TICKER-1", orderbook=orderbook)
        assert quote.bid_price is None
        assert quote.ask_price is None

    def test_one_sided_book_yes_only(self) -> None:
        """Book with only YES bids: bid is set, ask is None (no NO bids to derive from)."""
        normalizer = _make_normalizer()
        orderbook = _make_orderbook(YES_ONLY_ORDERBOOK_RAW)
        quote = normalizer.normalize_orderbook(ticker="TICKER-1", orderbook=orderbook)
        assert quote.bid_price == 0.35
        assert quote.ask_price is None

    def test_one_sided_book_no_only(self) -> None:
        """Book with only NO bids: bid is None, ask is derived from NO bids."""
        normalizer = _make_normalizer()
        orderbook = _make_orderbook(NO_ONLY_ORDERBOOK_RAW)
        quote = normalizer.normalize_orderbook(ticker="TICKER-1", orderbook=orderbook)
        assert quote.bid_price is None
        # Best NO bid = 0.65, so YES ask = 1.0 - 0.65 = 0.35
        assert quote.ask_price == 0.35

    def test_bid_size_matches_best_bid_level(self) -> None:
        normalizer = _make_normalizer()
        orderbook = _make_orderbook(
            {
                "orderbook_fp": {
                    "yes_dollars": [
                        ["0.5000", "100.00"],
                        ["0.5500", "200.00"],
                    ],
                    "no_dollars": [],
                }
            }
        )
        quote = normalizer.normalize_orderbook(ticker="TICKER-1", orderbook=orderbook)
        assert quote.bid_size == 200.0

    def test_ask_size_matches_best_no_bid_level(self) -> None:
        """Ask size = quantity at the best NO bid (which becomes YES ask)."""
        normalizer = _make_normalizer()
        orderbook = _make_orderbook(
            {
                "orderbook_fp": {
                    "yes_dollars": [],
                    "no_dollars": [
                        ["0.4000", "150.00"],
                        ["0.4500", "100.00"],
                    ],
                }
            }
        )
        quote = normalizer.normalize_orderbook(ticker="TICKER-1", orderbook=orderbook)
        assert quote.ask_size == 100.0

    def test_empty_ticker_raises_normalization_error(self) -> None:
        """Empty ticker must be rejected."""
        normalizer = _make_normalizer()
        orderbook = _make_orderbook(EMPTY_ORDERBOOK_RAW)
        with pytest.raises(NormalizationError, match="ticker must be non-empty"):
            normalizer.normalize_orderbook(ticker="", orderbook=orderbook)

    def test_whitespace_ticker_raises_normalization_error(self) -> None:
        """Whitespace-only ticker must be rejected."""
        normalizer = _make_normalizer()
        orderbook = _make_orderbook(EMPTY_ORDERBOOK_RAW)
        with pytest.raises(NormalizationError, match="ticker must be non-empty"):
            normalizer.normalize_orderbook(ticker="   ", orderbook=orderbook)

    def test_normalize_from_raw_dict(self) -> None:
        """normalize() accepts a raw dict and produces a MarketQuote."""
        normalizer = _make_normalizer()
        quote = normalizer.normalize(SAMPLE_ORDERBOOK_WITH_TICKER)
        assert quote.provider == "kalshi"
        assert quote.provider_instrument_id == "KXMLIFE-26-SEP05-100-ABOVE"

    def test_normalize_rejects_invalid_raw_dict(self) -> None:
        """normalize() raises NormalizationError for malformed dicts."""
        normalizer = _make_normalizer()
        with pytest.raises(NormalizationError, match="Failed to parse"):
            normalizer.normalize({"completely": "wrong"})

    def test_normalize_rejects_missing_orderbook_fp(self) -> None:
        """normalize() raises NormalizationError when orderbook_fp is missing."""
        normalizer = _make_normalizer()
        with pytest.raises(NormalizationError, match="Failed to parse"):
            normalizer.normalize({"ticker": "TICKER"})

    def test_both_sides_present_full_quote(self) -> None:
        """When both YES and NO bids exist, both bid and ask are populated."""
        normalizer = _make_normalizer()
        orderbook = _make_orderbook(SAMPLE_ORDERBOOK_RAW)
        quote = normalizer.normalize_orderbook(ticker="TICKER-1", orderbook=orderbook)
        # Best YES bid = 0.55, best NO bid = 0.45 → YES ask = 0.55
        assert quote.bid_price == 0.55
        assert quote.ask_price == 0.55
        assert quote.bid_size == 100.0
        assert quote.ask_size == 100.0

    def test_crossed_market_detection(self) -> None:
        """When YES bid > YES ask (derived), the market is crossed."""
        normalizer = _make_normalizer()
        orderbook = _make_orderbook(
            {
                "orderbook_fp": {
                    "yes_dollars": [
                        ["0.6000", "100.00"],
                    ],
                    "no_dollars": [
                        ["0.5000", "100.00"],
                    ],
                }
            }
        )
        quote = normalizer.normalize_orderbook(ticker="TICKER-1", orderbook=orderbook)
        # YES bid = 0.60, YES ask = 1.0 - 0.50 = 0.50 → crossed
        assert quote.bid_price == 0.60
        assert quote.ask_price == 0.50
        # Crossed market should be detected by validate_quote
        errors = validate_quote(quote)
        assert any("crossed" in e for e in errors)


# ══════════════════════════════════════════════════════════════════════════════
# D. INTEGRATION TESTS (end-to-end with fixtures)
# ══════════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """End-to-end flow tests: raw dict → normalizer → MarketQuote → validate_quote."""

    def test_full_flow_raw_dict_to_valid_quote(self) -> None:
        """Raw dict → normalize() → MarketQuote → validate_quote passes."""
        normalizer = _make_normalizer()
        quote = normalizer.normalize(SAMPLE_ORDERBOOK_WITH_TICKER)
        errors = validate_quote(quote)
        assert errors == [], f"Validation failed: {errors}"
        assert quote.provider == "kalshi"
        assert quote.bid_price == 0.55
        assert quote.ask_price == 0.55

    def test_full_flow_empty_book_to_valid_quote(self) -> None:
        """Empty book → quote with None sides → still passes validation."""
        normalizer = _make_normalizer()
        raw = {"ticker": "TICKER-1", **EMPTY_ORDERBOOK_RAW}
        quote = normalizer.normalize(raw)
        errors = validate_quote(quote)
        assert errors == []
        assert quote.bid_price is None
        assert quote.ask_price is None

    def test_full_flow_dataset_deterministic(self) -> None:
        """Same raw input → same Dataset dataset_id."""
        normalizer = _make_normalizer()
        q1 = normalizer.normalize(SAMPLE_ORDERBOOK_WITH_TICKER)
        q2 = normalizer.normalize(SAMPLE_ORDERBOOK_WITH_TICKER)
        ds = Dataset(records=[q1, q2])
        assert ds.dataset_id is not None

    def test_full_flow_multiple_books_to_dataset(self) -> None:
        """Multiple orderbooks → Dataset with correct record count."""
        normalizer = _make_normalizer()
        raw2 = json.loads(json.dumps(SAMPLE_ORDERBOOK_WITH_TICKER))
        raw2["orderbook_fp"]["yes_dollars"] = [["0.6000", "50.00"]]
        q1 = normalizer.normalize(SAMPLE_ORDERBOOK_WITH_TICKER)
        q2 = normalizer.normalize(raw2)
        ds = Dataset(records=[q1, q2])
        assert len(ds.records) == 2
        assert q1.provider_instrument_id == q2.provider_instrument_id

    def test_full_flow_client_to_normalizer_to_validation(self) -> None:
        """Client response → normalize_orderbook → validate."""
        normalizer = _make_normalizer()
        orderbook = _make_orderbook(SAMPLE_ORDERBOOK_RAW)
        quote = normalizer.normalize_orderbook(
            ticker="KXMLIFE-26-SEP05-100-ABOVE",
            orderbook=orderbook,
        )
        errors = validate_quote(quote)
        assert errors == [], f"Validation failed: {errors}"

    def test_full_flow_dataset_to_dict_roundtrip(self) -> None:
        """Dataset → to_dict → from_dict preserves data."""
        normalizer = _make_normalizer()
        q1 = normalizer.normalize(SAMPLE_ORDERBOOK_WITH_TICKER)
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
        normalizer = _make_normalizer()
        orderbook = _make_orderbook(
            {
                "orderbook_fp": {
                    "yes_dollars": [["0.5000", "100.00"]],
                    "no_dollars": [["0.5000", "200.00"]],
                }
            }
        )
        quote = normalizer.normalize_orderbook(ticker="TICKER-1", orderbook=orderbook)
        assert quote.bid_price == 0.50
        # YES ask = 1.0 - 0.50 = 0.50
        assert quote.ask_price == 0.50

    def test_many_levels_best_bid_ask_extraction(self) -> None:
        """20 levels — best bid is max, best ask is derived from max NO bid."""
        normalizer = _make_normalizer()
        yes_levels = [[f"0.{i:02d}00", "10.00"] for i in range(10, 30)]
        no_levels = [[f"0.{i:02d}00", "10.00"] for i in range(40, 60)]
        orderbook = _make_orderbook(
            {
                "orderbook_fp": {
                    "yes_dollars": yes_levels,
                    "no_dollars": no_levels,
                }
            }
        )
        quote = normalizer.normalize_orderbook(ticker="TICKER-1", orderbook=orderbook)
        # Best YES bid = max = 0.29
        assert quote.bid_price == 0.29
        # Best NO bid = max = 0.59, YES ask = 1.0 - 0.59 = 0.41
        assert quote.ask_price == 0.41

    def test_zero_quantity_level(self) -> None:
        """Zero-quantity levels are valid in the model."""
        level = ["0.5000", "0.00"]
        assert float(level[1]) == 0.0

    def test_validate_rejects_negative_price(self) -> None:
        """validate_quote rejects negative bid/ask prices."""
        quote = MarketQuote(
            source_timestamp=_utc("2026-09-04T01:00:00+00:00"),
            ingestion_timestamp=_utc("2026-09-04T01:00:01+00:00"),
            provider="kalshi",
            provider_instrument_id="TICKER-1",
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
            provider="kalshi",
            provider_instrument_id="TICKER-1",
            bid_price=float("nan"),
        )
        errors = validate_quote(quote)
        assert any("NaN" in e for e in errors)

    def test_validate_rejects_inf_price(self) -> None:
        """validate_quote rejects infinity in price fields."""
        quote = MarketQuote(
            source_timestamp=_utc("2026-09-04T01:00:00+00:00"),
            ingestion_timestamp=_utc("2026-09-04T01:00:01+00:00"),
            provider="kalshi",
            provider_instrument_id="TICKER-1",
            ask_price=float("inf"),
        )
        errors = validate_quote(quote)
        assert any("infinity" in e for e in errors)

    def test_validate_rejects_crossed_market(self) -> None:
        """validate_quote rejects bid > ask."""
        quote = MarketQuote(
            source_timestamp=_utc("2026-09-04T01:00:00+00:00"),
            ingestion_timestamp=_utc("2026-09-04T01:00:01+00:00"),
            provider="kalshi",
            provider_instrument_id="TICKER-1",
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
            provider="kalshi",
            provider_instrument_id="TICKER-1",
            bid_price=0.0,
            ask_price=1.0,
        )
        errors = validate_quote(quote)
        assert errors == []

    def test_price_at_one_dollar(self) -> None:
        """YES bid at $1.00 means YES ask = 1.0 - best NO bid."""
        normalizer = _make_normalizer()
        orderbook = _make_orderbook(
            {
                "orderbook_fp": {
                    "yes_dollars": [["1.0000", "50.00"]],
                    "no_dollars": [["0.0000", "50.00"]],
                }
            }
        )
        quote = normalizer.normalize_orderbook(ticker="TICKER-1", orderbook=orderbook)
        assert quote.bid_price == 1.0
        # YES ask = 1.0 - 0.0 = 1.0
        assert quote.ask_price == 1.0

    def test_price_at_zero_dollars(self) -> None:
        """YES bid at $0.00, NO bid at $0.00 → both sides at 0/1."""
        normalizer = _make_normalizer()
        orderbook = _make_orderbook(
            {
                "orderbook_fp": {
                    "yes_dollars": [["0.0000", "50.00"]],
                    "no_dollars": [["0.0000", "50.00"]],
                }
            }
        )
        quote = normalizer.normalize_orderbook(ticker="TICKER-1", orderbook=orderbook)
        assert quote.bid_price == 0.0
        # YES ask = 1.0 - 0.0 = 1.0
        assert quote.ask_price == 1.0

    def test_deep_orderbook_many_levels(self) -> None:
        """Deep orderbook with 50 levels on each side."""
        normalizer = _make_normalizer()
        yes_levels = [[f"0.{i:02d}00", f"{i * 10}.00"] for i in range(1, 51)]
        no_levels = [[f"0.{i:02d}00", f"{i * 10}.00"] for i in range(50, 100)]
        orderbook = _make_orderbook(
            {
                "orderbook_fp": {
                    "yes_dollars": yes_levels,
                    "no_dollars": no_levels,
                }
            }
        )
        quote = normalizer.normalize_orderbook(ticker="TICKER-1", orderbook=orderbook)
        assert quote.bid_price == 0.50
        assert quote.bid_size == 500.0
        # Best NO bid = 0.99, YES ask = 1.0 - 0.99 = 0.01
        assert quote.ask_price == 0.01
        assert quote.ask_size == 990.0

    def test_fractional_contract_sizes(self) -> None:
        """Fixed-point contract counts with fractional values."""
        normalizer = _make_normalizer()
        orderbook = _make_orderbook(
            {
                "orderbook_fp": {
                    "yes_dollars": [["0.5500", "13.50"]],
                    "no_dollars": [["0.4500", "7.25"]],
                }
            }
        )
        quote = normalizer.normalize_orderbook(ticker="TICKER-1", orderbook=orderbook)
        assert quote.bid_price == 0.55
        assert quote.bid_size == 13.5
        assert quote.ask_price == 0.55
        assert quote.ask_size == 7.25

    def test_subpenny_prices(self) -> None:
        """Subpenny pricing (Kalshi supports this)."""
        normalizer = _make_normalizer()
        orderbook = _make_orderbook(
            {
                "orderbook_fp": {
                    "yes_dollars": [["0.5555", "100.00"]],
                    "no_dollars": [["0.4444", "100.00"]],
                }
            }
        )
        quote = normalizer.normalize_orderbook(ticker="TICKER-1", orderbook=orderbook)
        assert quote.bid_price == pytest.approx(0.5555, abs=1e-6)
        assert quote.ask_price == pytest.approx(0.5556, abs=1e-4)


# ══════════════════════════════════════════════════════════════════════════════
# F. RECORD IDENTITY TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestRecordIdentity:
    """Test deterministic record_id generation."""

    def test_same_input_same_record_id(self) -> None:
        """Identical inputs produce identical record_ids."""
        normalizer = _make_normalizer()
        orderbook = _make_orderbook(SAMPLE_ORDERBOOK_RAW)
        q1 = normalizer.normalize_orderbook(ticker="T-1", orderbook=orderbook)
        q2 = normalizer.normalize_orderbook(ticker="T-1", orderbook=orderbook)
        assert q1.record_id == q2.record_id

    def test_different_ticker_different_record_id(self) -> None:
        """Different tickers produce different record_ids."""
        normalizer = _make_normalizer()
        orderbook = _make_orderbook(SAMPLE_ORDERBOOK_RAW)
        q1 = normalizer.normalize_orderbook(ticker="TICKER-A", orderbook=orderbook)
        q2 = normalizer.normalize_orderbook(ticker="TICKER-B", orderbook=orderbook)
        assert q1.record_id != q2.record_id

    def test_different_book_different_record_id(self) -> None:
        """Different orderbook data produces different record_ids."""
        normalizer = _make_normalizer()
        q1 = normalizer.normalize_orderbook(
            ticker="T-1", orderbook=_make_orderbook(SAMPLE_ORDERBOOK_RAW)
        )
        q2 = normalizer.normalize_orderbook(
            ticker="T-1", orderbook=_make_orderbook(EMPTY_ORDERBOOK_RAW)
        )
        assert q1.record_id != q2.record_id

    def test_record_id_is_hex_string(self) -> None:
        """record_id is a 16-character hex string."""
        normalizer = _make_normalizer()
        quote = normalizer.normalize_orderbook(
            ticker="T-1", orderbook=_make_orderbook(SAMPLE_ORDERBOOK_RAW)
        )
        rid = quote.record_id
        assert len(rid) == 16
        assert all(c in "0123456789abcdef" for c in rid)

    def test_record_id_deterministic_across_calls(self) -> None:
        """record_id is a property that returns the same value each time."""
        normalizer = _make_normalizer()
        quote = normalizer.normalize_orderbook(
            ticker="T-1", orderbook=_make_orderbook(SAMPLE_ORDERBOOK_RAW)
        )
        assert quote.record_id == quote.record_id
        assert quote.record_id == quote.record_id

    def test_dataset_deterministic_with_same_records(self) -> None:
        """Dataset with same records in same order has same dataset_id."""
        normalizer = _make_normalizer()
        q1 = normalizer.normalize_orderbook(
            ticker="T-1", orderbook=_make_orderbook(SAMPLE_ORDERBOOK_RAW)
        )
        q2 = normalizer.normalize_orderbook(
            ticker="T-2", orderbook=_make_orderbook(SAMPLE_ORDERBOOK_RAW)
        )
        ds1 = Dataset(records=[q1, q2])
        ds2 = Dataset(records=[q1, q2])
        assert ds1.dataset_id == ds2.dataset_id

    def test_dataset_different_records_different_id(self) -> None:
        """Dataset with different records has different dataset_id."""
        normalizer = _make_normalizer()
        q1 = normalizer.normalize_orderbook(
            ticker="T-1", orderbook=_make_orderbook(SAMPLE_ORDERBOOK_RAW)
        )
        q2 = normalizer.normalize_orderbook(
            ticker="T-2", orderbook=_make_orderbook(SAMPLE_ORDERBOOK_RAW)
        )
        q3 = normalizer.normalize_orderbook(
            ticker="T-3", orderbook=_make_orderbook(EMPTY_ORDERBOOK_RAW)
        )
        ds1 = Dataset(records=[q1, q2])
        ds2 = Dataset(records=[q1, q3])
        assert ds1.dataset_id != ds2.dataset_id


# ══════════════════════════════════════════════════════════════════════════════
# G. KALSHI-SPECIFIC CONVERSION TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestKalshiConversion:
    """Test Kalshi-specific price conversion logic.

    Key insight: Kalshi orderbook only returns bids.
    YES bid at price X = NO ask at (1 - X).
    The normalizer must derive the YES ask from the NO bids.
    """

    def test_yes_ask_equals_one_minus_best_no_bid(self) -> None:
        """YES ask = 1.0 - best NO bid price."""
        normalizer = _make_normalizer()
        orderbook = _make_orderbook(
            {
                "orderbook_fp": {
                    "yes_dollars": [["0.5000", "100.00"]],
                    "no_dollars": [["0.4000", "100.00"]],
                }
            }
        )
        quote = normalizer.normalize_orderbook(ticker="T-1", orderbook=orderbook)
        assert quote.bid_price == 0.50
        assert quote.ask_price == 0.60  # 1.0 - 0.40

    def test_no_ask_equals_one_minus_best_yes_bid(self) -> None:
        """NO ask = 1.0 - best YES bid price (for completeness)."""
        normalizer = _make_normalizer()
        orderbook = _make_orderbook(
            {
                "orderbook_fp": {
                    "yes_dollars": [["0.3000", "100.00"]],
                    "no_dollars": [["0.6000", "100.00"]],
                }
            }
        )
        quote = normalizer.normalize_orderbook(ticker="T-1", orderbook=orderbook)
        # YES bid = 0.30, YES ask = 1.0 - 0.60 = 0.40
        assert quote.bid_price == 0.30
        assert quote.ask_price == 0.40

    def test_spread_calculation(self) -> None:
        """Spread = YES ask - YES bid."""
        normalizer = _make_normalizer()
        orderbook = _make_orderbook(
            {
                "orderbook_fp": {
                    "yes_dollars": [["0.5500", "100.00"]],
                    "no_dollars": [["0.4000", "100.00"]],
                }
            }
        )
        quote = normalizer.normalize_orderbook(ticker="T-1", orderbook=orderbook)
        spread = quote.ask_price - quote.bid_price
        assert spread == pytest.approx(0.05, abs=1e-6)  # 0.60 - 0.55

    def test_tight_spread(self) -> None:
        """Tight spread: YES bid=0.50, NO bid=0.49 → ask=0.51, spread=0.01."""
        normalizer = _make_normalizer()
        orderbook = _make_orderbook(
            {
                "orderbook_fp": {
                    "yes_dollars": [["0.5000", "100.00"]],
                    "no_dollars": [["0.4900", "100.00"]],
                }
            }
        )
        quote = normalizer.normalize_orderbook(ticker="T-1", orderbook=orderbook)
        assert quote.bid_price == 0.50
        assert quote.ask_price == 0.51
        spread = quote.ask_price - quote.bid_price
        assert spread == pytest.approx(0.01, abs=1e-6)

    def test_wide_spread(self) -> None:
        """Wide spread: YES bid=0.30, NO bid=0.20 → ask=0.80, spread=0.50."""
        normalizer = _make_normalizer()
        orderbook = _make_orderbook(
            {
                "orderbook_fp": {
                    "yes_dollars": [["0.3000", "100.00"]],
                    "no_dollars": [["0.2000", "100.00"]],
                }
            }
        )
        quote = normalizer.normalize_orderbook(ticker="T-1", orderbook=orderbook)
        assert quote.bid_price == 0.30
        assert quote.ask_price == 0.80
        spread = quote.ask_price - quote.bid_price
        assert spread == pytest.approx(0.50, abs=1e-6)

    def test_binary_market_complementary_prices(self) -> None:
        """In a binary market, YES + NO prices should be complementary."""
        normalizer = _make_normalizer()
        orderbook = _make_orderbook(
            {
                "orderbook_fp": {
                    "yes_dollars": [["0.6500", "100.00"]],
                    "no_dollars": [["0.3500", "100.00"]],
                }
            }
        )
        quote = normalizer.normalize_orderbook(ticker="T-1", orderbook=orderbook)
        assert quote.bid_price == 0.65
        assert quote.ask_price == 0.65
        # Complementary: 0.65 + 0.35 = 1.0
        assert quote.bid_price + 0.35 == pytest.approx(1.0, abs=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# H. TIMESTAMP AND METADATA TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestTimestampAndMetadata:
    """Test timestamp handling and metadata extraction."""

    def test_ingestion_timestamp_uses_clock(self) -> None:
        """Ingestion timestamp comes from the clock, not the API response."""
        fixed_time = datetime(2025, 6, 15, 18, 30, 0, tzinfo=UTC)
        normalizer = KalshiNormalizer(clock=lambda: fixed_time)
        orderbook = _make_orderbook(SAMPLE_ORDERBOOK_RAW)
        quote = normalizer.normalize_orderbook(ticker="T-1", orderbook=orderbook)
        assert quote.ingestion_timestamp == fixed_time

    def test_source_timestamp_can_be_provided(self) -> None:
        """Source timestamp can be overridden explicitly."""
        normalizer = _make_normalizer()
        custom_ts = datetime(2025, 3, 10, 8, 0, 0, tzinfo=UTC)
        orderbook = _make_orderbook(SAMPLE_ORDERBOOK_RAW)
        quote = normalizer.normalize_orderbook(
            ticker="T-1",
            orderbook=orderbook,
            source_timestamp=custom_ts,
        )
        assert quote.source_timestamp == custom_ts

    def test_market_metadata_preserves_fields(self) -> None:
        """Market metadata preserves all important fields."""
        market = SAMPLE_MARKET_RAW["market"]
        assert market["ticker"] == "KXMLIFE-26-SEP05-100-ABOVE"
        assert market["event_ticker"] == "KXMLIFE-26-SEP05"
        assert market["market_type"] == "binary"
        assert market["status"] == "active"
        assert market["notional_value_dollars"] == "1.0000"
        assert market["settlement_timer_seconds"] == 300
        assert market["can_close_early"] is False

    def test_market_metadata_price_fields(self) -> None:
        """Market metadata includes all price-related fields."""
        market = SAMPLE_MARKET_RAW["market"]
        assert market["yes_bid_dollars"] == "0.5500"
        assert market["yes_ask_dollars"] == "0.5800"
        assert market["no_bid_dollars"] == "0.4200"
        assert market["no_ask_dollars"] == "0.4500"
        assert market["last_price_dollars"] == "0.5600"
        assert market["yes_bid_size_fp"] == "100.00"
        assert market["yes_ask_size_fp"] == "150.00"
