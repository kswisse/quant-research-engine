"""Arbitrage cost and execution-risk models for prediction markets.

Extends Phase 2.2 gross arbitrage detection with explicit execution cost
modeling and legging/exposure information.

This is a deterministic research calculation — it does NOT model:
- live trading
- wallet integration
- private keys
- transaction signing
- blockchain RPC
- order submission
- real-money execution
- cross-venue arbitrage
- real-time execution probability prediction

Architecture:
    ArbitrageOpportunity ─┐
                           ├→ evaluate_arbitrage_costs()
    ArbitrageCostModel   ─┘            ↓
                               NetArbitrageResult
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quant_engine.arbitrage.models import ArbitrageOpportunity


@dataclass(frozen=True)
class ArbitrageCostModel:
    """Immutable cost model for arbitrage execution.

    Represents caller-supplied assumptions about execution costs.
    All values are non-negative. Fee rates are proportions (e.g., 0.02 = 2%).

    Attributes:
        yes_fee_rate: Percentage of YES execution notional (0.0 to 1.0).
        no_fee_rate: Percentage of NO execution notional (0.0 to 1.0).
        fixed_cost: Generic fixed operational/transaction cost.
        settlement_cost: Optional explicit settlement cost.
    """

    yes_fee_rate: float = 0.0
    no_fee_rate: float = 0.0
    fixed_cost: float = 0.0
    settlement_cost: float = 0.0

    def __post_init__(self) -> None:
        """Validate cost model parameters."""
        # Check for NaN and infinity
        for field_name in ("yes_fee_rate", "no_fee_rate", "fixed_cost", "settlement_cost"):
            value = getattr(self, field_name)
            if math.isnan(value) or math.isinf(value):
                raise ValueError(f"{field_name} must be finite, got {value}")

        # Check for negative values
        if self.yes_fee_rate < 0:
            raise ValueError(f"yes_fee_rate must be >= 0, got {self.yes_fee_rate}")
        if self.no_fee_rate < 0:
            raise ValueError(f"no_fee_rate must be >= 0, got {self.no_fee_rate}")
        if self.fixed_cost < 0:
            raise ValueError(f"fixed_cost must be >= 0, got {self.fixed_cost}")
        if self.settlement_cost < 0:
            raise ValueError(f"settlement_cost must be >= 0, got {self.settlement_cost}")

        # Check fee rates are <= 1.0 (100%)
        if self.yes_fee_rate > 1.0:
            raise ValueError(f"yes_fee_rate must be <= 1.0, got {self.yes_fee_rate}")
        if self.no_fee_rate > 1.0:
            raise ValueError(f"no_fee_rate must be <= 1.0, got {self.no_fee_rate}")


@dataclass(frozen=True)
class NetArbitrageResult:
    """Result of evaluating arbitrage economics under execution costs.

    Preserves the distinction between gross execution, fees, other costs,
    and net result. Includes legging/exposure information.

    Attributes:
        provider: Provider identifier.
        yes_instrument_id: Provider's YES instrument identifier.
        no_instrument_id: Provider's NO instrument identifier.
        requested_size: Original requested number of pairs.
        yes_filled: Number of YES contracts actually filled.
        no_filled: Number of NO contracts actually filled.
        paired_size: Number of complete YES+NO pairs (min of fills).
        unpaired_yes_size: YES contracts filled but not part of a pair.
        unpaired_no_size: NO contracts filled but not part of a pair.
        gross_cost: Total acquisition cost (YES notional + NO notional).
        yes_fee: Fee charged on YES execution.
        no_fee: Fee charged on NO execution.
        fixed_cost: Fixed operational cost.
        settlement_cost: Settlement cost.
        total_cost: gross_cost + yes_fee + no_fee + fixed_cost + settlement_cost.
        guaranteed_payoff: Settlement payoff (paired_size × $1.00).
        gross_profit: guaranteed_payoff - gross_cost.
        net_profit: guaranteed_payoff - total_cost.
        gross_return: gross_profit / gross_cost. None if gross_cost == 0.
        net_return: net_profit / total_cost. None if total_cost == 0.
        fully_paired: True if yes_filled == no_filled.
        pairing_ratio: paired_size / requested_size. None if requested_size == 0.
    """

    provider: str
    yes_instrument_id: str
    no_instrument_id: str
    requested_size: float
    yes_filled: float
    no_filled: float
    paired_size: float
    unpaired_yes_size: float
    unpaired_no_size: float
    gross_cost: float
    yes_fee: float
    no_fee: float
    fixed_cost: float
    settlement_cost: float
    total_cost: float
    guaranteed_payoff: float
    gross_profit: float
    net_profit: float
    gross_return: float | None
    net_return: float | None
    fully_paired: bool
    pairing_ratio: float | None


def evaluate_arbitrage_costs(
    opportunity: ArbitrageOpportunity,
    cost_model: ArbitrageCostModel,
) -> NetArbitrageResult:
    """Evaluate arbitrage opportunity under execution costs.

    Takes a detected gross arbitrage opportunity and applies explicit
    cost assumptions to produce a net economic result with legging
    exposure information.

    The function operates on actual execution notionals from the
    opportunity (depth-aware), not quoted prices.

    Args:
        opportunity: Detected gross arbitrage opportunity from Phase 2.2.
        cost_model: Caller-supplied cost assumptions.

    Returns:
        NetArbitrageResult with full cost and exposure details.
    """
    # Extract execution information
    yes_filled = opportunity.yes_execution.filled_size
    no_filled = opportunity.no_execution.filled_size
    gross_cost = opportunity.total_cost

    # Paired quantity is the minimum of both fills
    paired_size = min(yes_filled, no_filled)

    # Unpaired exposure
    unpaired_yes_size = yes_filled - paired_size
    unpaired_no_size = no_filled - paired_size

    # Calculate fees based on actual execution notionals
    yes_fee = gross_cost * cost_model.yes_fee_rate if cost_model.yes_fee_rate > 0 else 0.0
    # Approximate NO fee from total cost minus YES cost
    yes_cost = opportunity.yes_execution.total_notional
    no_cost = gross_cost - yes_cost
    no_fee = no_cost * cost_model.no_fee_rate if cost_model.no_fee_rate > 0 else 0.0

    # Total cost including all fees
    total_cost = gross_cost + yes_fee + no_fee + cost_model.fixed_cost + cost_model.settlement_cost

    # Guaranteed payoff from binary settlement
    guaranteed_payoff = paired_size * 1.0  # $1 per contract

    # Profit calculations
    gross_profit = guaranteed_payoff - gross_cost
    net_profit = guaranteed_payoff - total_cost

    # Return calculations (None for zero denominators)
    gross_return = gross_profit / gross_cost if gross_cost > 0 else None
    net_return = net_profit / total_cost if total_cost > 0 else None

    # Legging metrics
    fully_paired = yes_filled == no_filled
    pairing_ratio = (
        paired_size / opportunity.requested_size
        if opportunity.requested_size > 0
        else None
    )

    return NetArbitrageResult(
        provider=opportunity.provider,
        yes_instrument_id=opportunity.yes_instrument_id,
        no_instrument_id=opportunity.no_instrument_id,
        requested_size=opportunity.requested_size,
        yes_filled=yes_filled,
        no_filled=no_filled,
        paired_size=paired_size,
        unpaired_yes_size=unpaired_yes_size,
        unpaired_no_size=unpaired_no_size,
        gross_cost=gross_cost,
        yes_fee=yes_fee,
        no_fee=no_fee,
        fixed_cost=cost_model.fixed_cost,
        settlement_cost=cost_model.settlement_cost,
        total_cost=total_cost,
        guaranteed_payoff=guaranteed_payoff,
        gross_profit=gross_profit,
        net_profit=net_profit,
        gross_return=gross_return,
        net_return=net_return,
        fully_paired=fully_paired,
        pairing_ratio=pairing_ratio,
    )
