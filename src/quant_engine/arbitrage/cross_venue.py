"""Cross-venue arbitrage detection for prediction markets.

Detects mechanically executable arbitrage opportunities between two venues
representing the same economic event/outcome.

This is a deterministic research calculation — it does NOT model:
- live trading
- order submission
- wallets
- private keys
- blockchain RPC
- authentication flows
- automatic execution
- cross-venue smart order routing
- real-money trading
- LLM-based market matching
- automatic fuzzy semantic event matching

Architecture:
    OrderBookSnapshot (buy venue) ─┐
                                   ├→ detect_cross_venue_arbitrage()
    OrderBookSnapshot (sell venue) ─┘           ↓
                                    CrossVenueOpportunity | None
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from quant_engine.order_book.execution import consume_book

if TYPE_CHECKING:
    from quant_engine.order_book.execution import ExecutionResult
    from quant_engine.order_book.models import OrderBookSnapshot


@dataclass(frozen=True)
class CrossVenueOpportunity:
    """A detected cross-venue arbitrage opportunity.

    Represents a mechanically executable opportunity to buy an outcome
    on one venue and sell the same outcome on another venue at a higher
    price, capturing the spread.

    This is a relative value trade, not a pure arbitrage, unless both
    venues share identical resolution mechanics and settlement timing.

    Attributes:
        outcome_label: Human-readable outcome description.
        buy_venue: Provider identifier for the buy side.
        sell_venue: Provider identifier for the sell side.
        buy_instrument_id: Provider's instrument identifier on buy venue.
        sell_instrument_id: Provider's instrument identifier on sell venue.
        requested_size: Original requested number of contracts.
        executable_size: Number of contracts actually profitable.
        buy_execution: Execution result for buying on buy_venue.
        sell_execution: Execution result for selling on sell_venue.
        gross_spread: sell_notional - buy_notional (positive = profitable).
        gross_return: gross_spread / buy_notional. None if buy_notional == 0.
        fully_executable: True if requested_size was fully profitable.
    """

    outcome_label: str
    buy_venue: str
    sell_venue: str
    buy_instrument_id: str
    sell_instrument_id: str
    requested_size: float
    executable_size: float
    buy_execution: ExecutionResult
    sell_execution: ExecutionResult
    gross_spread: float
    gross_return: float | None
    fully_executable: bool


def detect_cross_venue_arbitrage(
    buy_book: OrderBookSnapshot,
    sell_book: OrderBookSnapshot,
    requested_size: float,
    outcome_label: str = "outcome",
) -> CrossVenueOpportunity | None:
    """Detect cross-venue arbitrage opportunity.

    For a given outcome available on two venues, detects whether buying
    on one venue and selling on the other yields a positive spread.

    The detector uses mechanical execution to determine what quantity can
    actually be acquired and at what total cost. It accounts for order-book
    depth and partial fills on both venues.

    This function assumes that the two order books represent the same
    economic outcome. Whether they actually do is the caller's responsibility.

    Args:
        buy_book: Order book for the buy leg (consume asks).
        sell_book: Order book for the sell leg (consume bids).
        requested_size: Number of contracts to trade. Must be > 0 and finite.
        outcome_label: Human-readable outcome description.

    Returns:
        CrossVenueOpportunity if profitable, None otherwise.

    Raises:
        ValueError: If requested_size is <= 0, NaN, or infinity.
        ValueError: If buy_book and sell_book are from the same provider.
    """
    # Validate inputs
    if requested_size <= 0 or math.isnan(requested_size) or math.isinf(requested_size):
        raise ValueError(f"requested_size must be > 0 and finite, got {requested_size}")

    # Verify different providers (cross-venue by definition)
    if buy_book.provider == sell_book.provider:
        raise ValueError(
            f"Cross-venue arbitrage requires different providers, "
            f"got {buy_book.provider!r} for both"
        )

    # Execute both legs at the full requested size
    buy_exec = consume_book(buy_book, "buy", requested_size)
    sell_exec = consume_book(sell_book, "sell", requested_size)

    # The executable quantity is limited by both legs
    executable_size = min(buy_exec.filled_size, sell_exec.filled_size)

    if executable_size <= 0:
        return None

    # If the full requested size was not executable on both legs,
    # recompute for the exact executable quantity to get accurate economics
    if executable_size < requested_size:
        buy_exec = consume_book(buy_book, "buy", executable_size)
        sell_exec = consume_book(sell_book, "sell", executable_size)

    # Calculate gross spread
    gross_spread = sell_exec.total_notional - buy_exec.total_notional

    # Strictly positive spread required
    if gross_spread <= 0:
        return None

    # Return on buy-side capital
    gross_return = gross_spread / buy_exec.total_notional if buy_exec.total_notional > 0 else None

    return CrossVenueOpportunity(
        outcome_label=outcome_label,
        buy_venue=buy_book.provider,
        sell_venue=sell_book.provider,
        buy_instrument_id=buy_book.provider_instrument_id,
        sell_instrument_id=sell_book.provider_instrument_id,
        requested_size=requested_size,
        executable_size=executable_size,
        buy_execution=buy_exec,
        sell_execution=sell_exec,
        gross_spread=gross_spread,
        gross_return=gross_return,
        fully_executable=(executable_size == requested_size),
    )
