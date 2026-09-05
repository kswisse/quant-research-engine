"""Raw Polymarket API response models.

These represent the provider-specific schema. They are NOT canonical —
the normalizer converts these into MarketQuote records.

Do NOT add these models to quant_engine.market_data.
They live in quant_engine.providers.polymarket.

API Reference:
- CLOB API: https://clob.polymarket.com (order books, prices)
- Gamma API: https://gamma-api.polymarket.com (market metadata)

Timestamp formats:
- CLOB /book endpoint: timestamp is a string of epoch milliseconds (e.g., "1782753357257")
- Gamma API: No timestamps in market responses (metadata only)

Price/size encoding:
- All prices and sizes are strings (decimal representation)
- Prices are in range [0.00, 1.00] for binary markets
- Sizes are in outcome tokens (notional value)

Instrument identifiers:
- condition_id: Hex string identifying the market condition (e.g., "0x747dc...")
- token_id: Large integer string identifying a specific outcome token
- The Gamma API returns clobTokenIds as a JSON-encoded array string
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class OrderBookLevel(BaseModel):
    """Single price level in an order book.

    Attributes:
        price: Decimal string representation of the price (e.g., "0.45").
        size: Decimal string representation of the size (e.g., "100").
    """

    model_config = {"frozen": True}

    price: str
    size: str


class OrderBookSummary(BaseModel):
    """Raw order book response from CLOB API GET /book.

    Contains the full order book for a single outcome token, including
    all bid and ask levels, trading constraints, and metadata.

    Attributes:
        market: Condition ID (hex string) identifying the market.
        asset_id: Token ID (large integer string) for this outcome.
        timestamp: Epoch milliseconds as a string (e.g., "1782753357257").
        hash: Hash of the order book state for change detection.
        bids: List of bid levels sorted by price ascending (best bid last).
        asks: List of ask levels sorted by price descending (best ask last).
        min_order_size: Minimum order size as a decimal string.
        tick_size: Minimum price increment as a decimal string.
        neg_risk: Whether negative risk is enabled for this market.
        last_trade_price: Last traded price as a decimal string.
    """

    model_config = {"frozen": True}

    # Market identifiers
    market: str = Field(description="Condition ID (hex string)")
    asset_id: str = Field(description="Token ID (large integer string)")

    # Timestamp as epoch milliseconds string
    timestamp: str = Field(description="Epoch milliseconds as string")

    # Order book state hash for change detection
    hash: str = Field(description="Hash of order book state")

    # Price levels
    bids: list[OrderBookLevel] = Field(description="Bids sorted by price ascending")
    asks: list[OrderBookLevel] = Field(description="Asks sorted by price descending")

    # Trading constraints
    min_order_size: str = Field(description="Minimum order size (decimal string)")
    tick_size: str = Field(description="Minimum price increment (decimal string)")
    neg_risk: bool = Field(description="Negative risk flag")

    # Last trade
    last_trade_price: str = Field(description="Last traded price (decimal string)")


class MarketMetadata(BaseModel):
    """Raw market metadata from Gamma API.

    Contains market identification, question text, and token IDs for
    the YES and NO outcomes. The Gamma API does not include price data
    or order book information.

    Attributes:
        id: Market ID (numeric string).
        slug: URL-friendly market identifier.
        question: The market question text.
        condition_id: Condition ID (hex string) for the market.
        clob_token_ids: JSON-encoded array string of token IDs
                       (e.g., '["token_yes", "token_no"]').
    """

    model_config = {"frozen": True}

    id: str = Field(description="Market ID (numeric string)")
    slug: str | None = Field(default=None, description="URL-friendly market identifier")
    question: str | None = Field(default=None, description="Market question text")
    condition_id: str | None = Field(default=None, description="Condition ID (hex string)")
    clob_token_ids: str | None = Field(
        default=None,
        description="JSON-encoded array of token IDs for YES and NO outcomes",
    )


class EventMetadata(BaseModel):
    """Raw event metadata from Gamma API.

    An event groups one or more markets under a shared question set.
    A single-market event asks one yes/no question; a multi-market event
    splits a broader question into individual outcomes.

    Attributes:
        id: Event ID (numeric string).
        slug: URL-friendly event identifier.
        title: Event title text.
        markets: List of market objects in this event.
    """

    model_config = {"frozen": True}

    id: str = Field(description="Event ID (numeric string)")
    slug: str | None = Field(default=None, description="URL-friendly event identifier")
    title: str | None = Field(default=None, description="Event title text")
    markets: list[MarketMetadata] = Field(
        default_factory=list, description="Markets in this event"
    )


class PriceResponse(BaseModel):
    """Raw price response from CLOB API GET /price.

    Attributes:
        price: Best price for the requested side as a decimal string.
    """

    model_config = {"frozen": True}

    price: str = Field(description="Best price (decimal string)")


class MidpointResponse(BaseModel):
    """Raw midpoint response from CLOB API GET /midpoint.

    Attributes:
        mid: Midpoint price as a decimal string.
    """

    model_config = {"frozen": True}

    mid: str = Field(description="Midpoint price (decimal string)")


class SpreadResponse(BaseModel):
    """Raw spread response from CLOB API GET /spread.

    Attributes:
        spread: Bid-ask spread as a decimal string.
    """

    model_config = {"frozen": True}

    spread: str = Field(description="Bid-ask spread (decimal string)")


class LastTradePriceResponse(BaseModel):
    """Raw last trade price response from CLOB API GET /last-trade-price.

    Attributes:
        price: Last traded price as a decimal string.
        side: Order side of the last trade ("BUY" or "SELL").
    """

    model_config = {"frozen": True}

    price: str = Field(description="Last traded price (decimal string)")
    side: str = Field(description="Order side: BUY or SELL")


class ErrorResponse(BaseModel):
    """Raw error response from Polymarket APIs.

    Attributes:
        error: Human-readable error message.
        code: Machine-readable error code (when provided).
        retry_after_seconds: Seconds to wait before retrying (when provided).
    """

    model_config = {"frozen": True}

    error: str = Field(description="Human-readable error message")
    code: str | None = Field(default=None, description="Machine-readable error code")
    retry_after_seconds: int | None = Field(
        default=None, description="Seconds to wait before retrying"
    )
