# Same-Market Arbitrage Detection

Provider-independent detection of mathematically exploitable binary prediction-market pricing relationships.

## Purpose

Detect when YES and NO contracts from the same binary market can be acquired for less than the guaranteed settlement payoff of $1.00 per complete pair.

## Architecture

```
External Providers
        ↓
Market Data Adapters
        ↓
Canonical OrderBookSnapshot
        ↓
Order Book Calculations
        ↓
Mechanical Execution Simulator
        ↓
Same-Market Arbitrage Detector
        ↓
ArbitrageOpportunity
```

## Binary Market Mechanics

In a binary prediction market, exactly one outcome pays $1.00 per contract at settlement. Therefore:

```
1 YES + 1 NO = $1.00 guaranteed payoff
```

This creates an arbitrage opportunity when:

```
YES_ask + NO_ask < $1.00
```

## Quoted vs Executable Mispricing

### Quoted Mispricing

Top-of-book prices may show an attractive relationship:

```
YES best ask = 0.47
NO best ask  = 0.50

sum = 0.97 < 1.00
```

### Executable Mispricing

The requested quantity must actually be executable through the order books:

```
YES:
0.47 × 10
0.49 × 1000

NO:
0.50 × 5
0.52 × 1000
```

A 100-contract arbitrage is NOT available at 0.97. Only 5 complete pairs can be formed at the quoted prices.

## Algorithm

The detector uses mechanical execution to determine what quantity can actually be acquired:

1. Execute YES buy for requested size
2. Execute NO buy for requested size
3. `executable_size = min(yes_filled, no_filled)`
4. Recompute for exact executable quantity
5. Check if `total_cost < guaranteed_payoff`

## ArbitrageOpportunity

```python
@dataclass(frozen=True)
class ArbitrageOpportunity:
    provider: str
    yes_instrument_id: str
    no_instrument_id: str
    requested_size: float
    executable_size: float
    yes_execution: ExecutionResult
    no_execution: ExecutionResult
    total_cost: float          # YES notional + NO notional
    guaranteed_payoff: float   # executable_size × $1.00
    gross_profit: float        # guaranteed_payoff - total_cost
    gross_return: float | None # gross_profit / total_cost
    fully_executable: bool     # requested_size was fully profitable
```

## Profitability Condition

For a complete pair quantity `q`:

```
gross_profit(q) = q - YES_cost(q) - NO_cost(q)
```

A genuine gross arbitrage requires:

```
gross_profit(q) > 0
```

Zero-profit is excluded.

## Example

```
YES = 0.45
NO  = 0.50

Pair cost = 0.95
Settlement payoff = 1.00
Gross profit = 0.05 per complete pair
Gross return = 0.05 / 0.95 ≈ 5.26%
```

For larger quantities, deeper levels have worse prices:

```
YES:
0.45 × 100
0.48 × 100

NO:
0.50 × 100
0.51 × 100

100 pairs: cost = 95, profit = 5
200 pairs: cost = 194, profit = 6
```

## Partial Liquidity

When YES has more liquidity than NO:

```
YES can fill 100
NO can fill 60

executable_size = 60 complete pairs
```

The detector correctly handles asymmetric liquidity.

## Market Identity

The detector verifies:
- Same provider (YES and NO from same venue)
- Both books have asks (can buy from both sides)

The binary-market relationship (YES/NO pairing) is guaranteed by the provider adapter.

## What This Is NOT

This detector identifies **gross theoretical arbitrage**. It explicitly excludes:

- Exchange fees
- Gas / settlement costs
- Latency
- Fill probability
- Queue position
- Adverse selection
- Price movement during execution
- Live trading

## Scope Limitations

| In Scope | Out of Scope |
|----------|--------------|
| Same-market YES+NO | Cross-venue arbitrage |
| Binary markets | Multi-outcome markets |
| Gross theoretical profit | Net profit after fees |
| Static order-book analysis | Real-time detection |
| Mechanical execution | Live trading |

## Deferred Functionality

| Concept | Phase |
|---------|-------|
| Exchange fees | 2.3+ |
| Cross-venue arbitrage | 3.0+ |
| Multi-outcome markets | 3.0+ |
| Real-time detection | 2.3+ |
| Live trading | Never (research only) |
