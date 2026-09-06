"""Same-market binary prediction-market arbitrage detection.

Detects mathematically exploitable YES+NO pricing relationships from
static order-book snapshots, then estimates the mechanically executable
economics using the Phase 2.1 execution simulator.

This is a mechanical calculation — it does NOT model:
- exchange fees
- gas / settlement costs
- latency
- fill probability
- queue position
- adverse selection
- price movement during execution
- live trading

Architecture:
    OrderBookSnapshot (YES) ─┐
                             ├→ detect_same_market_arbitrage()
    OrderBookSnapshot (NO)  ─┘           ↓
                                  ArbitrageOpportunity | None
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from quant_engine.arbitrage.models import ArbitrageOpportunity
from quant_engine.order_book.execution import consume_book

if TYPE_CHECKING:
    from quant_engine.order_book.models import OrderBookSnapshot


def detect_same_market_arbitrage(
    yes_book: OrderBookSnapshot,
    no_book: OrderBookSnapshot,
    requested_size: float,
) -> ArbitrageOpportunity | None:
    """Detect same-market YES+NO arbitrage opportunity.

    For a binary market, holding 1 YES + 1 NO guarantees a $1.00 settlement
    payoff. If the combined acquisition cost is below $1.00, there is a
    gross arbitrage opportunity.

    The detector uses mechanical execution to determine what quantity can
    actually be acquired and at what total cost. It accounts for order-book
    depth and partial fills.

    Args:
        yes_book: Order book for the YES outcome.
        no_book: Order book for the NO outcome.
        requested_size: Requested number of complete YES+NO pairs.
            Must be > 0 and finite.

    Returns:
        ArbitrageOpportunity if a profitable quantity exists, None otherwise.

    Raises:
        ValueError: If requested_size is <= 0, NaN, or infinity.
        ValueError: If YES and NO books are from different providers.
    """
    # Validate inputs
    if requested_size <= 0 or math.isnan(requested_size) or math.isinf(requested_size):
        raise ValueError(f"requested_size must be > 0 and finite, got {requested_size}")

    # Verify same provider (binary-market relationship guaranteed by adapter)
    if yes_book.provider != no_book.provider:
        raise ValueError(
            f"YES and NO books must be from the same provider, "
            f"got {yes_book.provider!r} and {no_book.provider!r}"
        )

    # Execute both sides at the full requested size
    yes_exec = consume_book(yes_book, "buy", requested_size)
    no_exec = consume_book(no_book, "buy", requested_size)

    # The maximum quantity of complete pairs is the minimum filled on both sides
    executable_size = min(yes_exec.filled_size, no_exec.filled_size)

    if executable_size <= 0:
        return None

    # If the full requested size was not executable on both sides,
    # recompute for the exact executable quantity to get accurate economics
    if executable_size < requested_size:
        yes_exec = consume_book(yes_book, "buy", executable_size)
        no_exec = consume_book(no_book, "buy", executable_size)

    total_cost = yes_exec.total_notional + no_exec.total_notional
    guaranteed_payoff = executable_size * 1.0  # Binary: $1 per contract
    gross_profit = guaranteed_payoff - total_cost

    # Strictly positive profit required
    if gross_profit <= 0:
        return None

    gross_return = gross_profit / total_cost if total_cost > 0 else None

    return ArbitrageOpportunity(
        provider=yes_book.provider,
        yes_instrument_id=yes_book.provider_instrument_id,
        no_instrument_id=no_book.provider_instrument_id,
        requested_size=requested_size,
        executable_size=executable_size,
        yes_execution=yes_exec,
        no_execution=no_exec,
        total_cost=total_cost,
        guaranteed_payoff=guaranteed_payoff,
        gross_profit=gross_profit,
        gross_return=gross_return,
        fully_executable=(executable_size == requested_size),
    )
