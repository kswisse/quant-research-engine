# Order Book Research Model

Provider-independent order book depth representation for prediction market research.

## Why OrderBookSnapshot Exists

The existing `MarketQuote` is a canonical top-of-book observation — it captures the best bid and ask at a single point in time. Research on prediction markets requires depth-aware analysis: how much liquidity exists at each price level, how the book is shaped, and what quantities could be consumed at various price points.

`OrderBookSnapshot` provides this depth-aware representation without contaminating the canonical market-data layer with research-specific concerns.

## Architecture

```
Provider Data (Polymarket, Kalshi)
        ↓
Canonical Market Data (MarketQuote)
        ↓
Order Book Snapshot (OrderBookSnapshot)
        ↓
Pure Research Calculations (spread, depth, consumption)
        ↓
Future Execution / Arbitrage Research
```

## MarketQuote vs OrderBookSnapshot

| Aspect | MarketQuote | OrderBookSnapshot |
|--------|-------------|-------------------|
| Depth | Top-of-book only | Full depth |
| Fields | bid_price, ask_price, bid_size, ask_size | bids, asks (tuples of levels) |
| Purpose | Time-series analysis | Depth analysis |
| Use case | Spread tracking, mid-price | Liquidity analysis, execution modeling |
| Canonical | Yes | Research layer |

## OrderBookLevel

Immutable representation of one price level:

```python
OrderBookLevel(price=0.55, size=100.0)
```

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| price | float | 0.0 ≤ price ≤ 1.0 | Prediction market probability |
| size | float | size ≥ 0.0 | Number of contracts (not USD) |

### Price Convention

Prices are in the [0.00, 1.00] range, representing the probability-weighted price of the YES outcome. This matches both Polymarket and Kalshi conventions for binary markets.

Boundary prices (0.0 and 1.0) are valid. Subpenny pricing (e.g., 0.5555) is supported.

### Size Semantics

Size represents the number of contracts available at a price level. Each contract settles to $1.00 for binary markets. Size is NOT notional value — use `level.notional` for that.

## OrderBookSnapshot

Immutable snapshot of order book depth:

```python
OrderBookSnapshot(
    provider="polymarket",
    provider_instrument_id="tok-123",
    source_timestamp=datetime(2026, 3, 15, 9, 59, 58, tzinfo=UTC),
    ingestion_timestamp=datetime(2026, 3, 15, 10, 0, 0, tzinfo=UTC),
    source_timestamp_missing=False,
    bids=(OrderBookLevel(price=0.50, size=100.0),),
    asks=(OrderBookLevel(price=0.55, size=200.0),),
)
```

## Bid/Ask Ordering

**Bids** are sorted descending by price (highest first):

```
bids[0] = best bid (highest price)
bids[1] = second best bid
...
```

**Asks** are sorted ascending by price (lowest first):

```
asks[0] = best ask (lowest price)
asks[1] = second best ask
...
```

Ordering is canonicalized at construction time. Callers do not need to sort.

## Duplicate Price Levels

Duplicate prices are aggregated during construction:

```
Input:  [0.40 × 10, 0.40 × 20]
Output: [0.40 × 30]
```

This is deterministic and preserves all liquidity. The aggregation happens in the Pydantic model validator before the snapshot is constructed.

## Empty Sides

One-sided books are valid:

```python
# Bid-only (no asks)
OrderBookSnapshot(
    ...,
    bids=(OrderBookLevel(price=0.50, size=100.0),),
    asks=(),
)

# Ask-only (no bids)
OrderBookSnapshot(
    ...,
    bids=(),
    asks=(OrderBookLevel(price=0.55, size=100.0),),
)
```

Empty sides use empty tuples, consistent with `None` semantics in `MarketQuote`.

## Crossed Book Policy

A crossed book occurs when `best_bid > best_ask`. This is:

- **Allowed at construction** — the model preserves raw data
- **Rejected by `validate_snapshot()`** — following the existing validation pattern

This matches the `MarketQuote` pattern where crossed markets are caught by `validate_quote()`, not by the model constructor.

## Timestamp Semantics

Reuses Phase 1.4 semantics exactly:

| Field | Description |
|-------|-------------|
| source_timestamp | When the provider generated this snapshot (UTC) |
| ingestion_timestamp | When our system received it (UTC) |
| source_timestamp_missing | True when provider has no source timestamp |

When `source_timestamp_missing=True`:
- `source_timestamp` is set to `ingestion_timestamp`
- This is NOT the actual exchange-side event time
- Do NOT use for look-back windows or event alignment

## Snapshot Identity

`snapshot_id(snapshot)` computes a deterministic 16-character hexadecimal identity:

```
SHA-256(canonical JSON of snapshot contents) → first 16 hex chars
```

Canonical JSON includes:
- provider
- provider_instrument_id
- ingestion_timestamp
- bids (sorted by price descending)
- asks (sorted by price ascending)
- schema_version

When `source_timestamp_missing=True`, `source_timestamp` is excluded from the hash (same logic as `MarketQuote.record_id`).

Same logical content always produces the same `snapshot_id`.

## Calculations

### best_bid / best_ask

```python
best_bid(snapshot)  # → float | None
best_ask(snapshot)  # → float | None
```

Returns the best price on each side, or `None` if the side is empty.

### spread

```python
spread(snapshot)  # → float | None
```

Returns `best_ask - best_bid`. Returns `None` if either side is empty. Can be negative for crossed books.

### cumulative_depth

```python
cumulative_depth(snapshot, side="bid", levels=5)  # → list[DepthLevel]
```

Returns the first N levels with cumulative size and notional information:

```python
DepthLevel(
    price=0.50,
    size=100.0,
    cumulative_size=100.0,
    cumulative_notional=50.0,
)
```

For bids: starts from best bid (highest price) downward.
For asks: starts from best ask (lowest price) upward.

## Why Depth ≠ Guaranteed Execution

Order book depth represents **available liquidity**, not guaranteed execution. Real execution depends on:

- Queue position
- Latency
- Fill probability
- Cancellations
- Adverse selection
- Fees
- Slippage

These concerns are explicitly out of scope for this model. Future phases will model execution semantics.

## Provider Independence

The `order_book/` module contains zero provider-specific branching:

```python
# FORBIDDEN
if provider == "kalshi":
    ...
if provider == "polymarket":
    ...
```

Provider-specific conversion belongs in the adapter layer. The model knows only canonical market-data concepts.

## Mechanical Execution Simulator

The `consume_book()` function provides deterministic, provider-independent simulation of consuming displayed liquidity from a static order book.

### How It Works

For a **BUY** order:
- Consumes asks from lowest to highest price
- At each level: `filled = min(remaining, level.size)`
- Tracks per-level fills and total notional

For a **SELL** order:
- Consumes bids from highest to lowest price
- Same fill logic as BUY

### ExecutionFill

```python
@dataclass(frozen=True)
class ExecutionFill:
    price: float   # Price at which the fill occurred
    size: float    # Number of contracts filled
```

The `notional` property computes `price * size`.

### ExecutionResult

```python
@dataclass(frozen=True)
class ExecutionResult:
    side: Literal["buy", "sell"]
    requested_size: float      # Original request
    filled_size: float         # Actually filled
    remaining_size: float      # Not filled (requested - filled)
    total_notional: float      # Sum of fill notionals
    vwap: float | None         # Volume-weighted avg price (None if nothing fills)
    fills: tuple[ExecutionFill, ...]  # Per-level details
    fully_filled: bool         # True if completely filled
```

### VWAP Calculation

```
VWAP = total_notional / filled_size
```

VWAP is `None` when `filled_size == 0` (nothing fills).

### Partial Fills

When the book doesn't have enough depth:

```python
# Book has 100 contracts at 0.40
# Requesting 150 contracts

result = consume_book(book, "buy", 150.0)
assert result.filled_size == 100.0
assert result.remaining_size == 50.0
assert result.fully_filled is False
```

### Empty Books

Empty sides return unfilled results:

```python
# No asks available
result = consume_book(book, "buy", 100.0)
assert result.filled_size == 0.0
assert result.vwap is None
```

### Crossed Books

Crossed books are consumed mechanically. The simulator does NOT reject or repair them — it simply consumes the displayed liquidity at each price level.

### What This Is NOT

This simulator models consumption of displayed static liquidity. It is NOT:

- A realistic execution/fill-probability model
- A guarantee of real-world execution prices
- An accounting for queue position, latency, or adverse selection
- A fee-inclusive cost model

Future phases will model execution semantics beyond mechanical consumption.

## Deferred Functionality

The following are explicitly out of scope for Phase 2.1:

| Concept | Phase |
|---------|-------|
| Fee calculation | 2.2+ |
| Slippage model | 2.2+ |
| Fill probability | 2.2+ |
| Queue position | 2.2+ |
| YES/NO arbitrage | 2.2+ |
| Cross-venue matching | 2.2+ |
| Latency model | 2.1+ |
| Kelly sizing | 2.1+ |
| Real-time streaming | Separate module |
| Trade history | Separate module |
| Multi-outcome markets | Separate module |
| Scalar markets | Separate module |
