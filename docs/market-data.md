# Market Data Layer

Provider-independent canonical data models for quantitative research.

## Why a Canonical Layer Exists

External market data providers (Polymarket, Kalshi, etc.) each have their own
API schemas. The research engine must never depend on provider-specific formats.
The canonical layer ensures:

1. **Provider independence** — research code works with any data source
2. **Data integrity** — validation catches malformed data at ingestion
3. **Provenance** — every record traces back to its source
4. **Reproducibility** — deterministic record and dataset IDs
5. **Look-ahead protection** — source vs ingestion timestamps preserved

## Architecture

```
Provider Adapter (future)
        ↓
Raw Provider Record
        ↓
MarketDataNormalizer.normalize()
        ↓
MarketQuote (canonical, validated)
        ↓
validate_quote() / validate_time_order()
        ↓
Dataset (ordered collection)
        ↓
Backtest / Quant Systems
```

## Canonical Record Model

### MarketQuote

| Field | Type | Description |
|-------|------|-------------|
| source_timestamp | datetime (UTC, tz-aware) | When provider generated the quote |
| ingestion_timestamp | datetime (UTC, tz-aware) | When our system received it |
| provider | str | Provider identifier |
| provider_instrument_id | str | Provider's instrument ID |
| bid_price | float \| None | Best bid price (None = no bid) |
| bid_size | float \| None | Best bid size |
| ask_price | float \| None | Best ask price (None = no ask) |
| ask_size | float \| None | Best ask size |
| schema_version | str | Schema version (default "1") |
| record_id | str | Deterministic SHA-256 ID (computed) |

### Dataset

| Field | Type | Description |
|-------|------|-------------|
| schema_version | str | Schema version |
| records | list[MarketQuote] | Ordered records |
| dataset_id | str | Deterministic ID from ordered record IDs |

## Timestamp Policy

- **All timestamps are UTC, timezone-aware**
- Naive datetimes are rejected
- `source_timestamp`: when the market event occurred at the provider
- `ingestion_timestamp`: when our system received/recorded it
- These must never be confused — source time ≠ knowledge time

## Price/Size Semantics

- `float` representation for NumPy/Polars/Arrow compatibility
- Domain constraints enforced by validation, not the model
- `None` means "side not available" — never use 0, NaN, or sentinels
- Crossed markets (bid > ask) are rejected by default

## Record Identity

Each canonical record has a deterministic ID derived from:

```
SHA-256(provider + provider_instrument_id + source_timestamp + payload)
  → truncated to 16 hex chars
```

Same payload always produces the same record_id. This enables duplicate detection.

## Dataset Identity

```
SHA-256(schema_version + ordered record_ids)
  → truncated to 16 hex chars
```

Changing records or their order changes the dataset_id.

## Validation Rules

- `bid_price >= 0` if present
- `ask_price >= 0` if present
- `bid_size >= 0` if present
- `ask_size >= 0` if present
- `bid_price <= ask_price` if both present
- No NaN or infinity in numeric fields
- Timezone-aware timestamps required
- Non-empty provider and instrument IDs

## Duplicate Detection

Two records are duplicates if they share the same `record_id`. This means
same provider, same instrument, same source timestamp, and same payload.

## Out-of-Order Detection

`validate_time_order()` checks that source_timestamps are non-decreasing.
It reports violations without reordering.

## Normalization

Provider adapters implement `MarketDataNormalizer`:

```python
class MarketDataNormalizer(ABC):
    @abstractmethod
    def normalize(self, raw_record: dict[str, Any]) -> MarketQuote:
        ...
```

No provider-specific schema leaks into the canonical layer.

## Schema Versioning

`schema_version` field enables future schema evolution. Current version: `"1"`.

## Parquet/Arrow/Polars Compatibility

MarketQuote maps cleanly to tabular storage. Timestamps, prices, and sizes
are all compatible with Arrow types. Record IDs and provenance fields
preserve full traceability.

## Look-Ahead Protection

Source timestamps enable backtesting systems to enforce:

> Research code may only use information available at the simulated decision time.

Ingestion timestamps record when data was received — essential for
news/event research where latency matters.
