# Phase 1.0 — Market Data Architecture & Data Contract

## Objective

Establish the provider-independent market-data contract, validation, normalization abstractions, and local deterministic storage representation. No provider API clients.

## Architecture

```
External Market Data
        ↓
Provider Adapter (future)
        ↓
Raw Provider Record (future)
        ↓
MarketDataNormalizer (Protocol)
        ↓
MarketQuote (canonical record)
        ↓
validate_quote() / validate_time_order()
        ↓
Dataset (ordered collection with dataset_id)
        ↓
Backtest / Quant Systems
```

## File Structure

```
src/quant_engine/market_data/
├── __init__.py          # Public API
├── models.py            # MarketQuote, Dataset, ProviderInfo
├── validation.py        # validate_quote, validate_time_order, find_duplicates
├── normalization.py     # MarketDataNormalizer Protocol, ExampleNormalizer
└── errors.py            # MarketDataError hierarchy

tests/
└── test_market_data.py  # Comprehensive tests

docs/
├── market-data.md       # Full documentation
└── architecture.md      # Updated
```

## Key Design Decisions

- **Timestamps:** UTC, timezone-aware, microsecond precision via `datetime` with `tzinfo=UTC`
- **Prices/sizes:** `float` (compatible with NumPy/Polars/Arrow); domain constraints via validation
- **Record ID:** SHA-256 of canonical JSON of (provider, provider_instrument_id, source_timestamp, payload), truncated to 16 hex
- **Dataset ID:** SHA-256 of canonical JSON of (schema_version, ordered record IDs)
- **Missing values:** `None` for absent sides; never NaN/0/-1 sentinels
- **Schema version:** `"1"` string
- **Error hierarchy:** New errors under existing `QuantEngineError`
