# Arbitrage Cost & Execution-Risk Model

Extends Phase 2.2 gross arbitrage detection with explicit execution cost modeling and legging/exposure information.

## Purpose

Evaluate a detected gross arbitrage opportunity under explicit execution costs to produce a net economic result with legging exposure information.

## Architecture

```
ArbitrageOpportunity (Phase 2.2)
        ↓
ArbitrageCostModel (caller-supplied assumptions)
        ↓
evaluate_arbitrage_costs()
        ↓
NetArbitrageResult
```

## Why Gross Arbitrage Is Insufficient

Phase 2.2 detects gross arbitrage:

```
YES_cost(q) + NO_cost(q) < q
```

But real execution incurs:
- Trading fees (percentage of notional)
- Fixed transaction costs
- Settlement costs

An opportunity can be:
- Gross profitable: `gross_profit > 0`
- But net unprofitable: `net_profit <= 0`

Example:
```
gross_profit = 2.00
fees + other costs = 2.50
net_profit = -0.50
```

## Cost Model

```python
@dataclass(frozen=True)
class ArbitrageCostModel:
    yes_fee_rate: float = 0.0   # Percentage of YES notional (0.02 = 2%)
    no_fee_rate: float = 0.0    # Percentage of NO notional (0.03 = 3%)
    fixed_cost: float = 0.0     # Fixed operational cost
    settlement_cost: float = 0.0 # Settlement cost
```

All values are non-negative. Fee rates are proportions (0.0 to 1.0).

## Cost Components

### YES Trading Fee

Percentage of YES execution notional:
```
yes_fee = yes_cost × yes_fee_rate
```

### NO Trading Fee

Percentage of NO execution notional:
```
no_fee = no_cost × no_fee_rate
```

### Fixed Cost

Generic fixed operational/transaction cost.

### Settlement Cost

Optional explicit settlement cost.

## Total Cost Formula

```
total_cost =
    gross_cost
    + yes_fee
    + no_fee
    + fixed_cost
    + settlement_cost
```

## Profit Formulas

```
gross_profit = guaranteed_payoff - gross_cost
net_profit = guaranteed_payoff - total_cost
gross_return = gross_profit / gross_cost
net_return = net_profit / total_cost
```

For zero denominators: `return = None`.

## Legging Risk

The two legs are separate executions. The Phase 2.2 detector ensures equal fills by recomputing for the exact executable quantity, so legging doesn't occur in practice.

However, the cost model preserves the distinction for future extensions:

```
yes_filled: YES contracts actually filled
no_filled: NO contracts actually filled
paired_size: min(yes_filled, no_filled)
unpaired_yes_size: YES contracts not part of a pair
unpaired_no_size: NO contracts not part of a pair
```

## Pairing Metrics

```
pairing_ratio = paired_size / requested_size
fully_paired = (yes_filled == no_filled)
```

## Depth-Aware Execution

Costs are calculated on actual execution notionals from `consume_book()`, not quoted prices:

```
YES execution at 0.45 × 100 = 45.00 notional
NO execution at 0.50 × 100 = 50.00 notional
gross_cost = 95.00
```

## Example

```python
from quant_engine.arbitrage import (
    detect_same_market_arbitrage,
    evaluate_arbitrage_costs,
    ArbitrageCostModel,
)

# Detect gross arbitrage
yes_book = make_book(asks=[(0.45, 100)])
no_book = make_book(asks=[(0.50, 100)])
opp = detect_same_market_arbitrage(yes_book, no_book, 100)

# Apply cost model
cost_model = ArbitrageCostModel(
    yes_fee_rate=0.02,  # 2% fee on YES
    no_fee_rate=0.03,   # 3% fee on NO
    fixed_cost=1.00,    # $1 fixed cost
)

result = evaluate_arbitrage_costs(opp, cost_model)

print(f"Gross profit: ${result.gross_profit:.2f}")
print(f"Net profit: ${result.net_profit:.2f}")
print(f"YES fee: ${result.yes_fee:.2f}")
print(f"NO fee: ${result.no_fee:.2f}")
```

## Limitations

This is a deterministic research execution-cost model, not a live execution engine.

Explicitly excluded:
- Live trading
- Wallet integration
- Private keys
- Transaction signing
- Blockchain RPC
- Order submission
- Real-money execution
- Cross-venue arbitrage
- Real-time execution probability prediction

## Mathematical Invariants

For non-negative modeled costs:
```
net_profit <= gross_profit
paired_size <= requested_size
paired_size <= yes_filled
paired_size <= no_filled
unpaired_yes_size >= 0
unpaired_no_size >= 0
paired_size + unpaired_yes_size == yes_filled
paired_size + unpaired_no_size == no_filled
```
