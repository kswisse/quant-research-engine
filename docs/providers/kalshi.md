# Kalshi Data Adapter

Read-only adapter for Kalshi prediction market data.

## Status

**Phase 1.3 — Read-only data ingestion. No trading, no execution.**

## API Version & Base URL

| Item | Value |
|------|-------|
| API Version | Trade API v2 |
| Base URL | `https://external-api.kalshi.com/trade-api/v2` |
| Docs | https://docs.kalshi.com |

## Authentication

**No authentication required** for public market-data endpoints. All endpoints used by this adapter are public.

## API Endpoints Used

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/markets/{ticker}` | GET | — | Get market metadata by ticker |
| `/markets/{ticker}/orderbook` | GET | — | Get orderbook for a market |
| `/markets` | GET | — | List markets (paginated, filterable) |

### Not Used (requires authentication)

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/markets/orderbooks` | GET | Required | Batch orderbooks (up to 100 tickers) |

## Architecture

```
Kalshi API (Trade API v2)
        ↓
KalshiClient (provider-specific)
        ↓
Raw Kalshi Models (GetMarketOrderbookResponse, MarketObject, etc.)
        ↓
KalshiNormalizer
        ↓
MarketQuote (canonical, provider-independent)
        ↓
validate_quote() → Dataset
```

## Market Identity

| Concept | Value |
|---------|-------|
| Stable identifier | `ticker` (e.g., `"KXHIGHNY-24JAN01-T60"`) |
| Event grouping | `event_ticker` |
| Series grouping | `series_ticker` |

The `ticker` is the primary stable identifier for a market. It is human-readable, stable, and unique.

For `provider_instrument_id`: Use the market `ticker` directly.

## YES/NO Semantics

Kalshi markets are **binary** (YES/NO) contracts. The critical insight is the **duality principle**:

> A YES bid at price X is economically equivalent to a NO ask at (1 - X).

**The orderbook API only returns bids on each side — there are no asks.**

- `yes_dollars`: array of YES bid levels (people wanting to buy YES)
- `no_dollars`: array of NO bid levels (people wanting to buy NO)

**To derive the full bid/ask:**

| Quote Field | Derivation |
|-------------|------------|
| `bid_price` | Best YES bid (highest price in `yes_dollars`) |
| `ask_price` | 1.00 − best NO bid (from `no_dollars`) |
| `bid_size` | Count at best YES bid level |
| `ask_size` | Count at best NO bid level (same contracts, duality) |

## Price Units

| Aspect | Detail |
|--------|--------|
| Unit | US Dollars (not cents) |
| Range | 0.00 to 1.00 for binary markets |
| Format | Fixed-point decimal string (e.g., `"0.4200"`) |
| Conversion | `float("0.4200")` → `0.42` (no scaling needed) |

## Quantity Units

| Aspect | Detail |
|--------|--------|
| Unit | Contracts |
| Format | Fixed-point string with 2 decimal places |
| Example | `"100.00"` = 100 contracts |
| Fractional | Supported (e.g., `"2.50"` = 2.5 contracts) |

## Order Book Structure

```json
{
  "orderbook_fp": {
    "yes_dollars": [
      ["0.4100", "50.00"],
      ["0.4200", "100.00"],
      ["0.4300", "75.00"]
    ],
    "no_dollars": [
      ["0.5500", "200.00"],
      ["0.5600", "150.00"],
      ["0.5700", "80.00"]
    ]
  }
}
```

Each level is a 2-element string array: `[price_dollars, count_fp]`

- Element 0: Price in dollars (fixed-point string)
- Element 1: Contract count (fixed-point string)

Arrays are sorted ascending by price. **Best bid is the last element** (highest price).

## Top-of-Book Extraction

- Best YES bid = `max(yes_dollars, key=price)` (highest price)
- Best YES ask = `1.0 - max(no_dollars, key=price)` (derived from duality)
- Missing side → `None` (not 0, not NaN)

## Timestamps

| Aspect | Detail |
|--------|--------|
| Format | ISO 8601 datetime string |
| Timezone | UTC (e.g., `"2024-01-15T10:30:00Z"`) |
| Orderbook timestamps | **Not provided** — see `source_timestamp_missing` |

The orderbook endpoint does NOT return a timestamp. When `source_timestamp` is not provided, the normalizer uses `ingestion_timestamp` as the source timestamp and sets `source_timestamp_missing=True`.

### Semantic Limitation

When `source_timestamp_missing=True`:

- `source_timestamp` is set to `ingestion_timestamp` for storage
- This is **NOT** the actual exchange-side event time
- **Do NOT use for:**
  - Look-back window calculation
  - Event alignment across providers
  - Latency research
  - Historical event reconstruction
- Time-order validation skips these records
- Record ID excludes source_timestamp for stability

This is a known limitation of the Kalshi API. Future endpoints (e.g., trades, events) may expose timestamps.

## Pagination

Cursor-based pagination for `/markets`:

```
GET /markets?status=open&limit=100
→ { "markets": [...], "cursor": "abc123" }

GET /markets?status=open&limit=100&cursor=abc123
→ { "markets": [...], "cursor": "" }  // done
```

## Rate Limits

| Aspect | Detail |
|--------|--------|
| Model | Token bucket (10 tokens per request) |
| Basic tier | 200 tokens/sec → ~20 reads/sec |
| 429 response | HTTP 429 with `{"error": "too many requests"}` |
| Retry-After | **Not provided** — no `Retry-After` header |

No sophisticated throttling implemented. Use exponential backoff on 429.

## Error Handling

| Error | Exception |
|-------|-----------|
| HTTP error | `KalshiAPIError` (includes status code) |
| Timeout | `KalshiTimeoutError` |
| Connection failure | `KalshiConnectionError` |
| Malformed JSON | `KalshiClientError` |
| Invalid normalization | `NormalizationError` |

Error hierarchy: `KalshiClientError → MarketDataError → QuantEngineError`

## Deterministic Behavior

Same raw response → same normalized `MarketQuote` (except `ingestion_timestamp`). Use `clock` parameter for deterministic testing.

## Known Limitations

- Batch orderbook endpoint requires authentication (not implemented)
- Orderbook has no timestamp field (use ingestion time)
- No WebSocket/live streaming
- No trading, no execution, no wallet integration

## Example Normalized MarketQuote

```python
MarketQuote(
    source_timestamp=datetime(2025, 1, 1, 12, 0, tzinfo=UTC),
    source_timestamp_missing=True,  # Kalshi orderbook has no timestamp
    ingestion_timestamp=datetime(2025, 1, 1, 12, 0, tzinfo=UTC),
    provider="kalshi",
    provider_instrument_id="KXMLIFE-26-SEP05-100-ABOVE",
    bid_price=0.55,
    bid_size=100.0,
    ask_price=0.55,    # 1.0 - 0.45 (best NO bid)
    ask_size=100.0,    # count at best NO bid
    schema_version="1",
)
```
