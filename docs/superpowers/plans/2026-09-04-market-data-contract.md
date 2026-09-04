# Phase 1.0 — Market Data Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Establish provider-independent market-data contract, validation, normalization, and deterministic storage representation.

**Architecture:** Canonical `MarketQuote` records with deterministic IDs, validated by `validate_quote()` and `validate_time_order()`, organized into `Dataset` collections with dataset-level identity. `MarketDataNormalizer` Protocol for future provider adapters.

**Tech Stack:** Python 3.12+, Pydantic (frozen models), datetime (UTC), hashlib (SHA-256), pytest, ruff, mypy. Reuses existing `QuantEngineError` hierarchy.

## Global Constraints

- Platform: Windows 11 x64, Python 3.12.10
- No new dependencies beyond existing ones
- Do NOT implement: Polymarket/Kalshi clients, live ingestion, trading, databases, Docker, frontend
- Do NOT modify existing modules unless genuine bug found
- All code must pass: pytest, ruff check, ruff format --check, mypy
- Frozen Pydantic models, deterministic SHA-256 IDs, canonical JSON serialization

---

### Task 1: Errors, models, and core types

**Files:**
- Create: `src/quant_engine/market_data/__init__.py`
- Create: `src/quant_engine/market_data/errors.py`
- Create: `src/quant_engine/market_data/models.py`
- Test: `tests/test_market_data.py`

**Interfaces:**
- Consumes: `quant_engine.core.errors.QuantEngineError`
- Produces: `MarketDataError`, `InvalidTimestampError`, `InvalidQuoteError`, `DataIntegrityError`, `NormalizationError`, `MarketQuote`, `ProviderInfo`, `Dataset`

- [ ] **Step 1: Write failing tests for models**

Create `tests/test_market_data.py` with tests for MarketQuote, ProviderInfo, Dataset, and error hierarchy.

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement errors.py, models.py, __init__.py**

- [ ] **Step 4: Run tests to verify they pass**

- [ ] **Step 5: Commit**

---

### Task 2: Quote validation

**Files:**
- Create: `src/quant_engine/market_data/validation.py`
- Modify: `tests/test_market_data.py`

**Interfaces:**
- Consumes: `MarketQuote`
- Produces: `validate_quote()`, `validate_time_order()`, `find_duplicates()`

- [ ] **Step 1: Write failing tests for validation**

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement validation.py**

- [ ] **Step 4: Run tests to verify they pass**

- [ ] **Step 5: Commit**

---

### Task 3: Normalization abstraction

**Files:**
- Create: `src/quant_engine/market_data/normalization.py`
- Modify: `tests/test_market_data.py`

**Interfaces:**
- Consumes: `MarketQuote`
- Produces: `MarketDataNormalizer` Protocol, `ExampleNormalizer`

- [ ] **Step 1: Write failing tests for normalization**

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement normalization.py**

- [ ] **Step 4: Run tests to verify they pass**

- [ ] **Step 5: Commit**

---

### Task 4: Full validation, docs, and commit

**Files:**
- Create: `docs/market-data.md`
- Modify: `docs/architecture.md`
- Modify: `tests/test_market_data.py` (add edge cases, property tests)

- [ ] **Step 1: Add edge case and property-based tests**

- [ ] **Step 2: Run full test suite**

- [ ] **Step 3: Run ruff and mypy**

- [ ] **Step 4: Write docs/market-data.md**

- [ ] **Step 5: Update docs/architecture.md**

- [ ] **Step 6: Squash commits into single commit**

---

## Models

### ProviderInfo

```python
class ProviderInfo(BaseModel):
    model_config = {"frozen": True}
    provider: str                    # e.g. "polymarket", "kalshi"
    provider_instrument_id: str      # provider's own ID
    schema_version: str = "1"
```

### MarketQuote

```python
class MarketQuote(BaseModel):
    model_config = {"frozen": True}
    # Timestamps
    source_timestamp: datetime       # UTC, tz-aware — when provider generated event
    ingestion_timestamp: datetime    # UTC, tz-aware — when we received it
    # Instrument
    provider: str
    provider_instrument_id: str
    # Quote
    bid_price: float | None = None
    bid_size: float | None = None
    ask_price: float | None = None
    ask_size: float | None = None
    # Metadata
    schema_version: str = "1"
    record_id: str = ""              # computed property, SHA-256

    @property
    def record_id(self) -> str:
        """Deterministic ID from canonical payload."""
        ...
```

### Dataset

```python
class Dataset(BaseModel):
    model_config = {"frozen": True}
    schema_version: str = "1"
    records: list[MarketQuote]

    @property
    def dataset_id(self) -> str:
        """Deterministic ID from schema_version + ordered record IDs."""
        ...
```

## Validation Rules

- `bid_price >= 0` if not None
- `ask_price >= 0` if not None
- `bid_size >= 0` if not None
- `ask_size >= 0` if not None
- `bid_price <= ask_price` if both present (reject crossed)
- Reject NaN/inf in any numeric field
- Reject naive timestamps
- Reject empty provider/instrument IDs
- `validate_time_order(records)` → list of violations
- `find_duplicates(records)` → list of duplicate groups

## Record ID Derivation

```python
canonical = json.dumps({
    "provider": self.provider,
    "provider_instrument_id": self.provider_instrument_id,
    "source_timestamp": self.source_timestamp.isoformat(),
    "bid_price": self.bid_price,
    "bid_size": self.bid_size,
    "ask_price": self.ask_price,
    "ask_size": self.ask_size,
}, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
return hashlib.sha256(canonical.encode()).hexdigest()[:16]
```

## Dataset ID Derivation

```python
record_ids = [r.record_id for r in self.records]
canonical = json.dumps({
    "schema_version": self.schema_version,
    "record_ids": record_ids,
}, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
return hashlib.sha256(canonical.encode()).hexdigest()[:16]
```
