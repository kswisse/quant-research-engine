"""Polymarket public API client.

Read-only client for accessing Polymarket market data.
No authentication required for public endpoints.

Handles HTTP errors, malformed JSON, timeouts, and connection errors.
Does not retry indefinitely — bounded retry policy only.

API Reference:
- CLOB API: https://clob.polymarket.com (order books, prices)
- Gamma API: https://gamma-api.polymarket.com (market metadata)

Rate Limits (from official docs):
- CLOB /book: 1,500 req / 10s
- CLOB /price: 1,500 req / 10s
- CLOB /midpoint: 1,500 req / 10s
- CLOB /spread: 1,500 req / 10s
- Gamma /markets: 300 req / 10s
- Gamma /events: 500 req / 10s
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from quant_engine.market_data.errors import MarketDataError
from quant_engine.providers.polymarket.models import (
    ErrorResponse,
    EventMetadata,
    LastTradePriceResponse,
    MarketMetadata,
    MidpointResponse,
    OrderBookSummary,
    PriceResponse,
    SpreadResponse,
)


class PolymarketClientError(MarketDataError):
    """Base exception for Polymarket client errors."""


class PolymarketAPIError(PolymarketClientError):
    """Raised when the Polymarket API returns an error response.

    Attributes:
        status_code: HTTP status code.
        error_response: Parsed error response from the API.
    """

    def __init__(self, status_code: int, error_response: ErrorResponse) -> None:
        self.status_code = status_code
        self.error_response = error_response
        super().__init__(
            f"Polymarket API error {status_code}: {error_response.error}"
        )


class PolymarketConnectionError(PolymarketClientError):
    """Raised when unable to connect to the Polymarket API."""


class PolymarketTimeoutError(PolymarketClientError):
    """Raised when a request to the Polymarket API times out."""


class PolymarketClient:
    """Read-only client for Polymarket public APIs.

    Provides methods to fetch order books, prices, and market metadata
    from the Polymarket CLOB and Gamma APIs.

    Example:
        ```python
        client = PolymarketClient()

        # Fetch order book
        book = client.get_order_book(token_id="12345...")

        # Fetch best bid price
        price = client.get_price(token_id="12345...", side="BUY")

        # Fetch market metadata
        market = client.get_market(market_id="703257")
        ```
    """

    # API base URLs
    CLOB_BASE_URL = "https://clob.polymarket.com"
    GAMMA_BASE_URL = "https://gamma-api.polymarket.com"

    def __init__(self, timeout: float = 30.0) -> None:
        """Initialize the client.

        Args:
            timeout: Request timeout in seconds. Default is 30s.
        """
        self.timeout = timeout

    def _request(
        self,
        method: str,
        url: str,
        body: dict[str, Any] | list[Any] | None = None,
    ) -> Any:
        """Make an HTTP request and return parsed JSON.

        Args:
            method: HTTP method (GET, POST).
            url: Full URL to request.
            body: Optional JSON body for POST requests.

        Returns:
            Parsed JSON response.

        Raises:
            PolymarketAPIError: If the API returns an error response.
            PolymarketConnectionError: If unable to connect.
            PolymarketTimeoutError: If the request times out.
            PolymarketClientError: For other request failures.
        """
        headers = {"Accept": "application/json"}
        data = None

        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method=method,
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                response_data = response.read().decode("utf-8")
                return json.loads(response_data)
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8")
            try:
                error_json = json.loads(error_body)
                error_response = ErrorResponse(**error_json)
            except (json.JSONDecodeError, KeyError):
                error_response = ErrorResponse(
                    error=error_body or f"HTTP {exc.code}",
                    code=None,
                )
            raise PolymarketAPIError(exc.code, error_response) from exc
        except urllib.error.URLError as exc:
            if "timed out" in str(exc.reason).lower():
                raise PolymarketTimeoutError(
                    f"Request to {url} timed out after {self.timeout}s"
                ) from exc
            raise PolymarketConnectionError(
                f"Failed to connect to {url}: {exc.reason}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise PolymarketClientError(
                f"Invalid JSON response from {url}: {exc}"
            ) from exc

    # ── CLOB API Methods ──────────────────────────────────────────────

    def get_order_book(self, token_id: str) -> OrderBookSummary:
        """Fetch the full order book for an outcome token.

        Args:
            token_id: The outcome token ID (large integer string).

        Returns:
            OrderBookSummary with all bid/ask levels and metadata.

        Raises:
            PolymarketAPIError: If the token_id is invalid or not found.
        """
        url = f"{self.CLOB_BASE_URL}/book?token_id={token_id}"
        data = self._request("GET", url)
        return OrderBookSummary(**data)

    def get_price(self, token_id: str, side: str) -> PriceResponse:
        """Fetch the best price for a given side.

        For side="BUY", returns the lowest ask (price you'd pay).
        For side="SELL", returns the highest bid (price you'd receive).

        Args:
            token_id: The outcome token ID.
            side: Order side - "BUY" or "SELL".

        Returns:
            PriceResponse with the best price.

        Raises:
            PolymarketAPIError: If the token_id is invalid.
            ValueError: If side is not "BUY" or "SELL".
        """
        if side not in ("BUY", "SELL"):
            raise ValueError(f"side must be 'BUY' or 'SELL', got '{side}'")
        url = f"{self.CLOB_BASE_URL}/price?token_id={token_id}&side={side}"
        data = self._request("GET", url)
        return PriceResponse(**data)

    def get_midpoint(self, token_id: str) -> MidpointResponse:
        """Fetch the midpoint price (average of best bid and ask).

        Args:
            token_id: The outcome token ID.

        Returns:
            MidpointResponse with the midpoint price.
        """
        url = f"{self.CLOB_BASE_URL}/midpoint?token_id={token_id}"
        data = self._request("GET", url)
        return MidpointResponse(**data)

    def get_spread(self, token_id: str) -> SpreadResponse:
        """Fetch the bid-ask spread.

        Args:
            token_id: The outcome token ID.

        Returns:
            SpreadResponse with the spread.
        """
        url = f"{self.CLOB_BASE_URL}/spread?token_id={token_id}"
        data = self._request("GET", url)
        return SpreadResponse(**data)

    def get_last_trade_price(self, token_id: str) -> LastTradePriceResponse:
        """Fetch the last traded price and side.

        Args:
            token_id: The outcome token ID.

        Returns:
            LastTradePriceResponse with price and side.
        """
        url = f"{self.CLOB_BASE_URL}/last-trade-price?token_id={token_id}"
        data = self._request("GET", url)
        return LastTradePriceResponse(**data)

    def get_order_books(self, token_ids: list[str]) -> list[OrderBookSummary]:
        """Fetch order books for multiple tokens in one request.

        Maximum 500 items per request.

        Args:
            token_ids: List of outcome token IDs.

        Returns:
            List of OrderBookSummary objects.

        Raises:
            ValueError: If more than 500 token_ids provided.
        """
        if len(token_ids) > 500:
            raise ValueError(f"Maximum 500 token_ids per request, got {len(token_ids)}")
        url = f"{self.CLOB_BASE_URL}/books"
        body = [{"token_id": tid} for tid in token_ids]
        data = self._request("POST", url, body=body)
        return [OrderBookSummary(**item) for item in data]

    # ── Gamma API Methods ─────────────────────────────────────────────

    def get_market(self, market_id: str) -> MarketMetadata:
        """Fetch market metadata by ID.

        Args:
            market_id: The market ID (numeric string).

        Returns:
            MarketMetadata with market details.
        """
        url = f"{self.GAMMA_BASE_URL}/markets/{market_id}"
        data = self._request("GET", url)
        return MarketMetadata(**data)

    def get_market_by_slug(self, slug: str) -> MarketMetadata:
        """Fetch market metadata by slug.

        Args:
            slug: The market slug (URL-friendly identifier).

        Returns:
            MarketMetadata with market details.
        """
        url = f"{self.GAMMA_BASE_URL}/markets/slug/{slug}"
        data = self._request("GET", url)
        return MarketMetadata(**data)

    def get_event(self, event_id: str) -> EventMetadata:
        """Fetch event metadata by ID.

        Args:
            event_id: The event ID (numeric string).

        Returns:
            EventMetadata with event details and markets.
        """
        url = f"{self.GAMMA_BASE_URL}/events/{event_id}"
        data = self._request("GET", url)
        return EventMetadata(**data)

    def get_event_by_slug(self, slug: str) -> EventMetadata:
        """Fetch event metadata by slug.

        Args:
            slug: The event slug (URL-friendly identifier).

        Returns:
            EventMetadata with event details and markets.
        """
        url = f"{self.GAMMA_BASE_URL}/events/slug/{slug}"
        data = self._request("GET", url)
        return EventMetadata(**data)
