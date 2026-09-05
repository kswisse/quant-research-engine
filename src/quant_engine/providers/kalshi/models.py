"""Raw Kalshi API response models.

These represent the provider-specific schema. They are NOT canonical —
the normalizer converts these into MarketQuote records.

Do NOT add these models to quant_engine.market_data.
They live in quant_engine.providers.kalshi.

API Reference:
- Trade API v2: https://docs.kalshi.com
- Base URL: https://external-api.kalshi.com/trade-api/v2

Timestamp formats:
- ISO 8601 datetime strings (e.g., "2024-01-15T10:30:00Z")

Price/size encoding:
- Prices are fixed-point dollar strings in [0.00, 1.00] range
- Sizes are fixed-point contract count strings (e.g., "100.00")
- Each orderbook level is a 2-element string array: [price, count]

Instrument identifiers:
- ticker: Stable market identifier (e.g., "KXHIGHNY-24JAN01-T60")

Order book semantics (critical):
- The orderbook only returns BIDS on each side (YES and NO)
- There are NO explicit asks
- YES ask = 1.00 - best NO bid (binary market duality)
- NO ask = 1.00 - best YES bid
"""

from __future__ import annotations

from pydantic import BaseModel


class OrderBookLevel(BaseModel):
    """Single price level in a Kalshi order book.

    Kalshi returns levels as 2-element string arrays [price, count].
    This model wraps that into a typed structure.

    Attributes:
        price: Price in dollars as a fixed-point string (e.g., "0.4200").
        count: Contract count as a fixed-point string (e.g., "100.00").
    """

    model_config = {"frozen": True}

    price: str
    count: str


class OrderBookCountFp(BaseModel):
    """Kalshi order book response (fixed-point variant).

    Contains YES bid levels and NO bid levels. There are NO explicit
    asks — the ask side is derived via binary market duality:
        YES ask = 1.00 - best NO bid
        NO ask = 1.00 - best YES bid

    Arrays are sorted ascending by price. Best bid is the LAST element.

    Attributes:
        yes_dollars: YES bid levels as [[price, count], ...].
        no_dollars: NO bid levels as [[price, count], ...].
    """

    model_config = {"frozen": True}

    yes_dollars: list[list[str]]
    no_dollars: list[list[str]]


class GetMarketOrderbookResponse(BaseModel):
    """Response from GET /markets/{ticker}/orderbook.

    Attributes:
        orderbook_fp: The order book with fixed-point price/count levels.
    """

    model_config = {"frozen": True}

    orderbook_fp: OrderBookCountFp


class MarketObject(BaseModel):
    """Kalshi market metadata object.

    Represents the full metadata for a single binary (or scalar) market.

    Attributes:
        ticker: Stable market identifier.
        event_ticker: Parent event ticker.
        market_type: "binary" or "scalar".
        status: Market status (active, closed, etc.).
        created_time: ISO 8601 creation timestamp.
        updated_time: ISO 8601 last update timestamp.
        open_time: ISO 8601 market open timestamp.
        close_time: ISO 8601 market close timestamp.
    """

    model_config = {"frozen": True}

    ticker: str
    event_ticker: str
    market_type: str
    yes_sub_title: str = ""
    no_sub_title: str = ""
    created_time: str
    updated_time: str
    open_time: str = ""
    close_time: str = ""
    latest_expiration_time: str = ""
    settlement_timer_seconds: int = 0
    status: str = ""
    notional_value_dollars: str = "1.0000"
    yes_bid_dollars: str = ""
    yes_ask_dollars: str = ""
    no_bid_dollars: str = ""
    no_ask_dollars: str = ""
    yes_bid_size_fp: str = ""
    yes_ask_size_fp: str = ""
    last_price_dollars: str = ""
    previous_yes_bid_dollars: str = ""
    previous_yes_ask_dollars: str = ""
    previous_price_dollars: str = ""
    volume_fp: str = ""
    volume_24h_fp: str = ""
    open_interest_fp: str = ""
    result: str = ""
    can_close_early: bool = False
    expiration_value: str = ""
    rules_primary: str = ""
    rules_secondary: str = ""
    price_level_structure: list[str] | None = None
    price_ranges: list[str] | None = None
    title: str | None = None
    subtitle: str | None = None
    expected_expiration_time: str | None = None
    settlement_value_dollars: str | None = None
    settlement_ts: str | None = None


class GetMarketsResponse(BaseModel):
    """Response from GET /markets.

    Attributes:
        markets: List of market objects.
        cursor: Pagination cursor (empty string = no more pages).
    """

    model_config = {"frozen": True}

    markets: list[MarketObject]
    cursor: str = ""


class GetMarketResponse(BaseModel):
    """Response from GET /markets/{ticker}.

    Attributes:
        market: The market object.
    """

    model_config = {"frozen": True}

    market: MarketObject


class ErrorResponse(BaseModel):
    """Kalshi API error response.

    Attributes:
        code: Machine-readable error code.
        message: Human-readable error message.
        details: Additional details (optional).
        service: Service name (optional).
    """

    model_config = {"frozen": True}

    code: str | None = None
    message: str = ""
    details: str | None = None
    service: str | None = None
