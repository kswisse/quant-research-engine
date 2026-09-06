"""Arbitrage detection models for prediction markets.

Provides deterministic, provider-independent representation of
detected same-market arbitrage opportunities.

This is a mechanical calculation — it does NOT model:
- exchange fees
- gas / settlement costs
- latency
- fill probability
- queue position
- adverse selection
- price movement during execution
- live trading
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quant_engine.order_book.execution import ExecutionResult


@dataclass(frozen=True)
class ArbitrageOpportunity:
    """A detected same-market YES+NO arbitrage opportunity.

    Represents a mechanically executable opportunity to buy both YES and NO
    contracts from the same binary market for less than the guaranteed
    settlement payoff of $1.00 per complete pair.

    Attributes:
        provider: Provider identifier (e.g. "polymarket", "kalshi").
        yes_instrument_id: Provider's YES instrument identifier.
        no_instrument_id: Provider's NO instrument identifier.
        requested_size: Original requested number of pairs.
        executable_size: Number of complete YES+NO pairs actually profitable.
        yes_execution: Execution result for buying YES contracts.
        no_execution: Execution result for buying NO contracts.
        total_cost: Total acquisition cost (YES notional + NO notional).
        guaranteed_payoff: Settlement payoff (executable_size × $1.00).
        gross_profit: Guaranteed payoff minus total cost.
        gross_return: Gross profit divided by total cost. None if total_cost == 0.
        fully_executable: True if requested_size was fully profitable.
    """

    provider: str
    yes_instrument_id: str
    no_instrument_id: str
    requested_size: float
    executable_size: float
    yes_execution: ExecutionResult
    no_execution: ExecutionResult
    total_cost: float
    guaranteed_payoff: float
    gross_profit: float
    gross_return: float | None
    fully_executable: bool
