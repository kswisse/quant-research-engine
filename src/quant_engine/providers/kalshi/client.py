"""Kalshi public API client.

Read-only client for accessing Kalshi market data.
No authentication required for public endpoints.

Handles HTTP errors, malformed JSON, timeouts, and connection errors.
Does not retry indefinitely — bounded retry policy only.

API Reference:
- Trade API v2: https://docs.kalshi.com
- Base URL: https://external-api.kalshi.com/trade-api/v2

Rate Limits:
- Token bucket model (10 tokens per request)
- Basic tier: 200 tokens/sec → ~20 reads/sec
- Returns HTTP 429 on rate limit (no Retry-After header)

Public endpoints (no auth required):
- GET /markets — list markets (paginated)
- GET /markets/{ticker} — get market by ticker
- GET /markets/{ticker}/orderbook — get orderbook for a market
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from quant_engine.market_data.errors import MarketDataError
from quant_engine.providers.kalshi.models import (
    ErrorResponse,
    GetMarketOrderbookResponse,
    GetMarketResponse,
    GetMarketsResponse,
)


class KalshiClientError(MarketDataError):
    """Base exception for Kalshi client errors."""


class KalshiAPIError(KalshiClientError):
    """Raised when the Kalshi API returns an error response.

    Attributes:
        status_code: HTTP status code.
        error_response: Parsed error response from the API.
    """

    def __init__(self, status_code: int, error_response: ErrorResponse) -> None:
        self.status_code = status_code
        self.error_response = error_response
        super().__init__(f"Kalshi API error {status_code}: {error_response.message}")


class KalshiConnectionError(KalshiClientError):
    """Raised when unable to connect to the Kalshi API."""


class KalshiTimeoutError(KalshiClientError):
    """Raised when a request to the Kalshi API times out."""


class KalshiClient:
    """Read-only client for Kalshi public APIs.

    Provides methods to fetch order books and market metadata
    from the Kalshi Trade API v2.

    Example:
        ```python
        client = KalshiClient()

        # Fetch order book
        book = client.get_orderbook(ticker="KXHIGHNY-24JAN01-T60")

        # Fetch market metadata
        market = client.get_market(ticker="KXHIGHNY-24JAN01-T60")
        ```
    """

    BASE_URL = "https://external-api.kalshi.com/trade-api/v2"

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
            KalshiAPIError: If the API returns an error response.
            KalshiConnectionError: If unable to connect.
            KalshiTimeoutError: If the request times out.
            KalshiClientError: For other request failures.
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
                    message=error_body or f"HTTP {exc.code}",
                    code=None,
                )
            raise KalshiAPIError(exc.code, error_response) from exc
        except urllib.error.URLError as exc:
            if "timed out" in str(exc.reason).lower():
                raise KalshiTimeoutError(
                    f"Request to {url} timed out after {self.timeout}s"
                ) from exc
            raise KalshiConnectionError(f"Failed to connect to {url}: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise KalshiClientError(f"Invalid JSON response from {url}: {exc}") from exc

    def get_orderbook(self, ticker: str) -> GetMarketOrderbookResponse:
        """Fetch the order book for a market.

        Args:
            ticker: The market ticker (e.g., "KXHIGHNY-24JAN01-T60").

        Returns:
            GetMarketOrderbookResponse with YES/NO bid levels.

        Raises:
            KalshiAPIError: If the ticker is invalid or not found.
        """
        url = f"{self.BASE_URL}/markets/{ticker}/orderbook"
        data = self._request("GET", url)
        return GetMarketOrderbookResponse(**data)

    def get_market(self, ticker: str) -> GetMarketResponse:
        """Fetch market metadata by ticker.

        Args:
            ticker: The market ticker (e.g., "KXHIGHNY-24JAN01-T60").

        Returns:
            GetMarketResponse with market details.

        Raises:
            KalshiAPIError: If the ticker is invalid or not found.
        """
        url = f"{self.BASE_URL}/markets/{ticker}"
        data = self._request("GET", url)
        return GetMarketResponse(**data)

    def get_markets(
        self,
        status: str | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> GetMarketsResponse:
        """Fetch a list of markets (paginated).

        Args:
            status: Optional filter by market status (e.g., "active").
            cursor: Pagination cursor from a previous response.
            limit: Number of results per page (max 1000).

        Returns:
            GetMarketsResponse with market list and cursor.
        """
        params: list[str] = []
        if status is not None:
            params.append(f"status={status}")
        if cursor is not None:
            params.append(f"cursor={cursor}")
        if limit is not None:
            params.append(f"limit={limit}")

        url = f"{self.BASE_URL}/markets"
        if params:
            url += "?" + "&".join(params)

        data = self._request("GET", url)
        return GetMarketsResponse(**data)
