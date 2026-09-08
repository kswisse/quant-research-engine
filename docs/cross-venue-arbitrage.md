# Cross-Venue Arbitrage Detection

Detects mechanically executable arbitrage opportunities between two venues representing the same economic event/outcome.

## Purpose

Identify when an outcome can be bought cheaply on one venue and sold expensively on another, capturing the price spread.

## Architecture

```
OrderBookSnapshot (buy venue) ─┐
                               ├→ detect_cross_venue_arbitrage()
OrderBookSnapshot (sell venue) ─┘           ↓
                                CrossVenueOpportunity | None
                                        ↓
                                ArbitrageCostModel
                                        ↓
                                NetCrossVenueResult
```

## Cross-Venue Arbitrage Mechanics

The canonical example:

```
Venue A YES ask = 0.40
Venue B YES bid = 0.45
```

Buy at Venue A, sell at Venue B:

```
gross_spread = sell_proceeds - buy_cost
             = (0.45 × q) - (0.40 × q)
             = 0.05 × q
```

## Directionality

Supports both directions:

```
BUY A → SELL B  (when A ask < B bid)
BUY B → SELL A  (when B ask < A bid)
```

The detector evaluates both directions where market data permits.

## Depth-Aware Execution

Uses `consume_book()` for both legs:

- **Buy leg**: Consumes asks from lowest to highest
- **Sell leg**: Consumes bids from highest to lowest

Example:

```
Venue A:
ASK
0.40 × 100
0.41 × 100

Venue B:
BID
0.45 × 50
0.44 × 100
```

Requested: 150

Executable quantity is constrained by both venues.

## Executable Quantity

```
executable_size = min(buy_filled, sell_filled)
```

Only paired quantity represents the intended arbitrage.

## Exposure

```
unpaired_buy = buy_filled - executable_size
unpaired_sell = sell_filled - executable_size
```

Unpaired exposure represents directional position, not arbitrage profit.

## Gross Economics

```
gross_cost = buy_execution.total_notional
gross_proceeds = sell_execution.total_notional
gross_spread = gross_proceeds - gross_cost
gross_return = gross_spread / gross_cost
```

For zero denominator: `return = None`.

## Cost Model

Extends `ArbitrageCostModel` with cross-venue fields:

```python
ArbitrageCostModel(
    buy_fee_rate=0.02,     # 2% fee on buy execution
    sell_fee_rate=0.03,    # 3% fee on sell execution
    fixed_cost=1.00,       # Fixed operational cost
    settlement_cost=0.50,  # Settlement cost
    transfer_cost=0.25,    # Cross-venue transfer cost
)
```

## Net Economics

```
total_cost = gross_cost + buy_fee + sell_fee + fixed + settlement + transfer
net_spread = gross_proceeds - total_cost
net_return = net_spread / total_cost
```

## Market Identity Assumption

Phase 2.4 assumed that the two venue instruments have already been validated as the same economic market and outcome.

**Phase 2.5 update:** The detector now accepts an optional `MarketMapping` parameter for explicit identity validation.

```python
def detect_cross_venue_arbitrage(
    buy_book: OrderBookSnapshot,
    sell_book: OrderBookSnapshot,
    requested_size: float,
    outcome_label: str = "outcome",
    mapping: MarketMapping | None = None,
) -> CrossVenueOpportunity | None:
```

### With Mapping

When a `MarketMapping` is provided:
- The detector validates that order books match the mapping's provider/instrument identity
- The mapping is carried through to `CrossVenueOpportunity` and `NetCrossVenueResult`
- The outcome label is taken from the mapping (not the caller)

### Without Mapping (Backward Compatible)

When `mapping=None`:
- Backward compatible with Phase 2.4 behavior
- No identity validation is performed
- The caller is responsible for ensuring the markets are equivalent
- An absent/unvalidated mapping must never silently establish economic equivalence

### Why Mapping Is Required

Without explicit mapping, cross-venue arbitrage detection operates on unvalidated identity. This can produce false arbitrage signals when:
- Two markets have similar titles but resolve on different events
- Resolution sources differ (e.g., AP vs Reuters)
- Settlement terms differ (e.g., binary vs proportional)
- Expiry times differ
- One market is void/cancelled while the other is active

See [Market Identity & Resolution Semantics](market-identity.md) for details.

## Limitations

This is a deterministic research simulation, not a live execution engine.

Explicitly excluded:
- Live trading
- Order submission
- Wallets
- Private keys
- Blockchain RPC
- Authentication flows
- Automatic execution
- Cross-venue smart order routing
- Real-money trading
- LLM-based market matching
- Automatic fuzzy semantic event matching

## Research Value

Cross-venue arbitrage is a **relative value trade**, not a pure arbitrage, unless both venues share identical resolution mechanics and settlement timing.

Key risks to document:
- Resolution divergence
- Settlement timing mismatch
- Transfer latency
- Liquidity evaporation during execution

## Deferred Functionality

| Concept | Phase |
|---------|-------|
| Market identity resolution | 2.5+ |
| Settlement risk modeling | 2.5+ |
| Transfer latency modeling | 3.0+ |
| Live execution | Never |
