# Market Identity & Resolution Semantics

Phase 2.5 — Provider-independent market identity for cross-venue arbitrage detection.

## Purpose

Establish a domain foundation for distinguishing "similar contracts" from "economically equivalent contracts" by adding explicit market identity models, resolution/settlement/expiry compatibility validation, and a validated mapping layer that prevents cross-venue arbitrage detection from operating on unvalidated identity.

## Architecture

```
MarketIdentity (provider A) ─┐
                              ├→ MarketMapping (validates compatibility)
MarketIdentity (provider B) ─┘
         │
         ▼
detect_cross_venue_arbitrage()
         │
         ▼
CrossVenueOpportunity (carries mapping)
         │
         ▼
NetCrossVenueResult (carries mapping)
```

## Domain Models

### EventIdentity

Immutable identity for a prediction market event.

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | `str` | Provider-independent event identifier |
| `title` | `str` | Human-readable event title |
| `category` | `str` | Optional event category |
| `boundary` | `EventBoundary` | Temporal boundaries |
| `schema_version` | `str` | Schema version for forward compatibility |

### EventBoundary

Temporal boundaries for an event.

| Field | Type | Description |
|-------|------|-------------|
| `close_time` | `datetime` | When betting closes (UTC) |
| `expiry_time` | `datetime` | When the market expires (UTC) |

### OutcomeIdentity

Immutable identity for a specific outcome.

| Field | Type | Description |
|-------|------|-------------|
| `outcome_id` | `str` | Provider-independent outcome identifier |
| `label` | `str` | Human-readable outcome label |
| `description` | `str` | Detailed outcome description |

### ResolutionRule

Immutable resolution rule for a market.

| Field | Type | Description |
|-------|------|-------------|
| `source` | `str` | Resolution data source (e.g. "associated-press") |
| `method` | `str` | Resolution method (e.g. "official-certification") |
| `authority` | `str` | Resolution authority (e.g. "usc") |
| `is_definitive` | `bool` | Whether this source provides definitive resolution |

### ResolutionState

Enum representing the current settlement status of a market.

| Value | Description |
|-------|-------------|
| `RESOLVED` | Market has been officially resolved |
| `UNRESOLVED` | Market is still open / awaiting resolution |
| `VOID` | Market was cancelled, voided, or declared invalid |

**Compatibility rules:**
- `VOID` vs `RESOLVED` → **incompatible** (cannot be economically equivalent)
- `VOID` vs `UNRESOLVED` → **incompatible** (cannot be economically equivalent)
- `UNRESOLVED` vs `RESOLVED` → **compatible** (market may have resolved on one venue)

### SettlementTerms

Immutable settlement terms for a market.

| Field | Type | Description |
|-------|------|-------------|
| `payout_type` | `str` | Type of payout (e.g. "binary", "range") |
| `payout_cap` | `float` | Maximum payout per contract |
| `settlement_formula` | `str` | How payouts are calculated |
| `settlement_currency` | `str` | Currency for settlement (e.g. "USD", "USDC") |

### TemporalScope

Immutable temporal scope for market evaluation.

| Field | Type | Description |
|-------|------|-------------|
| `evaluation_time` | `datetime` | When the market evaluates the outcome (UTC) |
| `is_point_in_time` | `bool` | True for snapshot evaluation, False for path-dependent |
| `start_time` | `datetime \| None` | Start of evaluation window (for path-dependent) |

### MarketIdentity

Immutable composite market identity. Combines all the above into a single canonical identity.

| Field | Type | Description |
|-------|------|-------------|
| `provider` | `str` | Provider identifier (e.g. "polymarket", "kalshi") |
| `provider_market_id` | `str` | Provider's market identifier |
| `event` | `EventIdentity` | Event identity |
| `outcome` | `OutcomeIdentity` | Outcome identity |
| `resolution` | `ResolutionRule` | Resolution rule |
| `resolution_state` | `ResolutionState` | Whether resolved, unresolved, or void |
| `settlement` | `SettlementTerms` | Settlement terms |
| `temporal_scope` | `TemporalScope` | Temporal evaluation scope |
| `schema_version` | `str` | Schema version for forward compatibility |

**Key property:** `canonical_identity` — Deterministic SHA-256 hash derived from semantic fields only. Provider-specific identifiers are EXCLUDED. Two markets with identical semantic content from different providers produce the same `canonical_identity`.

## MarketMapping

Immutable mapping between two MarketIdentity instances.

| Field | Type | Description |
|-------|------|-------------|
| `source` | `MarketIdentity` | The source market identity |
| `target` | `MarketIdentity` | The target market identity |
| `relationship` | `str` | "same_market" or "opposite_outcome" |
| `schema_version` | `str` | Schema version for forward compatibility |

**Key property:** `mapping_id` — Deterministic SHA-256 hash derived from source/target canonical identities and relationship type.

### Explicit/Manual Mapping

Mappings are **curated by humans or a mapping service**, NOT inferred from market data. This ensures economic equivalence is never assumed from unvalidated identity.

### Construction-Time Validation

When a `MarketMapping` is constructed with `relationship="same_market"`, the following validations are performed:

1. **Event identity** — Same `event_id`
2. **Outcome identity** — Same `outcome_id`
3. **Resolution compatibility** — Same `source`, `method`, `authority`
4. **Resolution state compatibility** — VOID cannot match RESOLVED/UNRESOLVED
5. **Settlement compatibility** — Same formula, payout cap, type, currency
6. **Temporal scope** — Same evaluation time, same point-in-time flag, same start_time (for path-dependent markets)
7. **Event boundary** — Same close time, same expiry time

### Why Title Similarity Is Insufficient

Two markets with similar titles (e.g., "Will X win?" vs "X victory?") may:
- Resolve on different events
- Use different resolution sources
- Have different settlement terms
- Expire at different times

**Exact matching on semantic fields is required.** Title/description similarity is never used.

### Why `mapping=None` Does NOT Imply Economic Equivalence

When `mapping=None` is passed to `detect_cross_venue_arbitrage()`:
- The detector operates in legacy mode without identity validation
- No claim of economic equivalence is made
- The caller is responsible for ensuring the markets are equivalent
- An absent/unvalidated mapping must never silently establish economic equivalence

## Cross-Venue Integration

### detect_cross_venue_arbitrage()

```python
def detect_cross_venue_arbitrage(
    buy_book: OrderBookSnapshot,
    sell_book: OrderBookSnapshot,
    requested_size: float,
    outcome_label: str = "outcome",
    mapping: MarketMapping | None = None,
) -> CrossVenueOpportunity | None:
```

When a `mapping` is provided:
- The detector validates that order books match the mapping's provider/instrument identity
- The mapping is carried through to `CrossVenueOpportunity` and `NetCrossVenueResult`

When `mapping=None`:
- Backward compatible with Phase 2.4 behavior
- No identity validation is performed

### Provider-Specific vs Canonical Identity

| Concept | Scope | Example |
|---------|-------|---------|
| `provider` | Provider-specific | "polymarket" |
| `provider_market_id` | Provider-specific | "0x123abc" |
| `canonical_identity` | Provider-independent | SHA-256 hash of semantic fields |

The `canonical_identity` intentionally EXCLUDES provider-specific identifiers. Two markets with identical semantic content from different providers produce the same `canonical_identity`.

## Error Hierarchy

```
QuantEngineError
  └── MarketIdentityError
        ├── IncompatibleOutcomeError
        ├── IncompatibleResolutionError
        ├── IncompatibleResolutionStateError
        └── IncompatibleSettlementError
```

## Deterministic IDs

All IDs use SHA-256 truncated to 16 hex characters:
- `canonical_identity` on `MarketIdentity`
- `mapping_id` on `MarketMapping`

Canonical JSON is produced with:
- `sort_keys=True`
- `separators=(",", ":")`
- `ensure_ascii=True`
