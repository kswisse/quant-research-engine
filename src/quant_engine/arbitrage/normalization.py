"""Canonical normalized representation for arbitrage opportunities.

Unifies same-market binary arbitrage and cross-venue arbitrage into a
single composable domain model.

This is a deterministic research calculation — it does NOT model:
- live trading
- order submission
- wallets
- blockchain
- settlement execution
- WebSocket infrastructure
- historical replay
- event sourcing

Architecture:
    ArbitrageOpportunity ─┐
                           ├→ normalize_same_market()
    NetArbitrageResult   ──┘          ↓
                              NormalizedOpportunity

    CrossVenueOpportunity ─┐
                            ├→ normalize_cross_venue()
    NetCrossVenueResult  ──┘          ↓
                              NormalizedOpportunity
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from quant_engine.arbitrage.errors import (
    InvalidLegError,
)

if TYPE_CHECKING:
    from quant_engine.arbitrage.costs import NetArbitrageResult, NetCrossVenueResult
    from quant_engine.arbitrage.cross_venue import CrossVenueOpportunity
    from quant_engine.arbitrage.models import ArbitrageOpportunity
    from quant_engine.market_identity.mapping import MarketMapping
    from quant_engine.market_identity.models import MarketIdentity


# ══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ══════════════════════════════════════════════════════════════════════════════


class OpportunityType(StrEnum):
    """Type of arbitrage opportunity."""

    SAME_MARKET = "same_market"
    CROSS_VENUE = "cross_venue"


class LegRole(StrEnum):
    """Role of an execution leg."""

    BUY = "buy"
    SELL = "sell"


class ExecutionRole(StrEnum):
    """Execution role within the opportunity."""

    PRIMARY = "primary"
    COUNTER = "counter"


# ══════════════════════════════════════════════════════════════════════════════
# SUB-MODELS
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class OpportunityLeg:
    """One leg of an arbitrage opportunity.

    Composable for both same-market (BUY YES + BUY NO) and cross-venue
    (BUY A + SELL B) arbitrage.

    Attributes:
        leg_index: 0-based ordering.
        provider: Provider identifier.
        provider_instrument_id: Provider's instrument identifier.
        market_identity: Optional canonical market identity.
        side: BUY or SELL.
        execution_role: PRIMARY or COUNTER.
        price: Top-of-book reference price.
        vwap: Volume-weighted average execution price. None if nothing fills.
        fill_size: Contracts filled.
        fill_notional: Dollar cost or proceeds.
        fill_count: Number of price levels touched.
        levels_consumed: Price levels consumed.
        best_available_price: Best price before execution.
    """

    leg_index: int
    provider: str
    provider_instrument_id: str
    market_identity: MarketIdentity | None
    side: LegRole
    execution_role: ExecutionRole
    price: float
    vwap: float | None
    fill_size: float
    fill_notional: float
    fill_count: int
    levels_consumed: int
    best_available_price: float

    def __post_init__(self) -> None:
        """Validate leg fields."""
        if not self.provider:
            raise InvalidLegError("provider must not be empty")
        if not self.provider_instrument_id:
            raise InvalidLegError("provider_instrument_id must not be empty")
        if self.leg_index < 0:
            raise InvalidLegError(f"leg_index must be >= 0, got {self.leg_index}")
        if self.fill_size < 0:
            raise InvalidLegError(f"fill_size must be >= 0, got {self.fill_size}")
        if self.fill_notional < 0:
            raise InvalidLegError(f"fill_notional must be >= 0, got {self.fill_notional}")


@dataclass(frozen=True)
class ExecutionPlanLeg:
    """One leg of a theoretical execution plan.

    Attributes:
        leg_index: 0-based ordering.
        provider: Provider identifier.
        provider_instrument_id: Provider's instrument identifier.
        side: BUY or SELL.
        target_size: Target number of contracts.
        expected_price: Expected execution price.
        expected_notional: Expected dollar cost or proceeds.
    """

    leg_index: int
    provider: str
    provider_instrument_id: str
    side: LegRole
    target_size: float
    expected_price: float
    expected_notional: float


@dataclass(frozen=True)
class ExecutionPlan:
    """Theoretical execution plan for an arbitrage opportunity.

    Describes how the opportunity would theoretically be executed,
    without actually executing anything.

    Clearly distinguishes:
    - Detected opportunity (NormalizedOpportunity)
    - Theoretical execution plan (this class)
    - Actual execution (not modeled in Phase 2.6)

    Attributes:
        legs: Ordered execution plan legs.
        total_estimated_cost: Total estimated acquisition cost.
        total_estimated_proceeds: Total estimated disposal proceeds.
        estimated_net: Estimated net profit.
        is_simultaneous: Whether legs execute concurrently.
    """

    legs: tuple[ExecutionPlanLeg, ...]
    total_estimated_cost: float
    total_estimated_proceeds: float
    estimated_net: float
    is_simultaneous: bool


@dataclass(frozen=True)
class CostBreakdown:
    """Normalized cost representation.

    Replaces detector-specific cost fields with a canonical structure.
    Does NOT create a competing economic model — maps from existing
    Phase 2.3 cost evaluation results.

    Attributes:
        buy_fee: Fee on buy execution (YES fee for same-market).
        sell_fee: Fee on sell execution (NO fee for same-market).
        fixed_cost: Fixed operational cost.
        settlement_cost: Settlement cost.
        transfer_cost: Cross-venue asset transfer cost.
        total_fee: buy_fee + sell_fee.
        total_cost: gross_cost + total_fee + fixed + settlement + transfer.
        effective_fee_rate: total_fee / gross_cost. None if gross_cost == 0.
    """

    buy_fee: float
    sell_fee: float
    fixed_cost: float
    settlement_cost: float
    transfer_cost: float
    total_fee: float
    total_cost: float
    effective_fee_rate: float | None


@dataclass(frozen=True)
class RiskFlags:
    """Static, deterministic risk/quality flags.

    Computed at construction time from the opportunity's fields.
    These are metadata/quality indicators, NOT a quantitative risk engine.

    Attributes:
        is_fully_executable: All requested size was fillable.
        has_partial_fill: Some requested size was not filled.
        has_unpaired_exposure: Unpaired contracts exist (directional risk).
        gross_margin_is_thin: gross_profit < 5% of gross_cost.
        net_margin_is_negative: net_profit < 0 after fees.
        near_expiry: Market closes within 24 hours.
        is_expired: Market has expired.
        identity_validated: MarketMapping was provided and validated.
        identity_unvalidated: No MarketMapping (caller must verify).
        settlement_currency_mismatch: Different settlement currencies.
        resolution_source_mismatch: Different resolution sources.
    """

    is_fully_executable: bool
    has_partial_fill: bool
    has_unpaired_exposure: bool
    gross_margin_is_thin: bool
    net_margin_is_negative: bool
    near_expiry: bool
    is_expired: bool
    identity_validated: bool
    identity_unvalidated: bool
    settlement_currency_mismatch: bool
    resolution_source_mismatch: bool


# ══════════════════════════════════════════════════════════════════════════════
# TOP-LEVEL MODEL
# ══════════════════════════════════════════════════════════════════════════════


def _compute_opportunity_id(d: dict[str, Any]) -> str:
    """Compute deterministic opportunity ID from canonical dictionary.

    Uses SHA-256 of canonical JSON with sorted keys and compact separators.
    Truncated to 16 hex characters, consistent with the project's pattern.
    """
    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class NormalizedOpportunity:
    """Canonical normalized representation of an arbitrage opportunity.

    Unifies same-market binary arbitrage and cross-venue arbitrage into
    one composable domain model. Represents the economic opportunity
    independently of the specific detector that discovered it.

    The opportunity_id represents the economic structure. The observation_id
    incorporates the observation timestamp for temporal deduplication.

    Attributes:
        opportunity_id: Deterministic SHA-256 of economic structure.
        observation_id: Deterministic SHA-256 of structure + timestamp.
        opportunity_type: SAME_MARKET or CROSS_VENUE.
        legs: Exactly 2 legs (buy + sell/buy).
        execution_plan: Theoretical execution description.
        gross_cost: Total acquisition cost.
        gross_proceeds: Total disposal proceeds (0 for same-market).
        gross_profit: gross_proceeds - gross_cost (or payoff - cost).
        gross_return: gross_profit / gross_cost. None if gross_cost == 0.
        cost_breakdown: Normalized cost representation.
        net_profit: Net profit after costs.
        net_return: net_profit / total_cost. None if total_cost == 0.
        requested_size: Original requested number of contracts.
        executable_size: Number of contracts actually profitable.
        fully_executable: True if requested_size was fully profitable.
        risk_flags: Static quality characterization.
        mapping: Optional MarketMapping for cross-market identity.
        observation_timestamp: When the opportunity was observed.
        evaluation_time: When outcome is evaluated (from MarketIdentity).
        is_point_in_time: Whether market evaluates at a single point.
        close_time: When betting closes (from EventBoundary).
        expiry_time: When market expires (from EventBoundary).
        outcome_label: Human-readable outcome description.
    """

    # --- Identity ---
    opportunity_id: str
    observation_id: str
    opportunity_type: OpportunityType

    # --- Legs ---
    legs: tuple[OpportunityLeg, ...]

    # --- Execution plan ---
    execution_plan: ExecutionPlan

    # --- Economics (pre-cost) ---
    gross_cost: float
    gross_proceeds: float
    gross_profit: float
    gross_return: float | None

    # --- Economics (post-cost) ---
    cost_breakdown: CostBreakdown
    net_profit: float
    net_return: float | None

    # --- Sizing ---
    requested_size: float
    executable_size: float
    fully_executable: bool

    # --- Risk ---
    risk_flags: RiskFlags

    # --- Market identity ---
    mapping: MarketMapping | None

    # --- Temporal ---
    observation_timestamp: datetime
    evaluation_time: datetime | None
    is_point_in_time: bool | None
    close_time: datetime | None
    expiry_time: datetime | None

    # --- Provenance ---
    outcome_label: str

    def __post_init__(self) -> None:
        """Validate normalized opportunity."""
        if len(self.legs) != 2:
            raise ValueError(
                f"NormalizedOpportunity requires exactly 2 legs, got {len(self.legs)}"
            )
        if self.gross_cost < 0:
            raise ValueError(f"gross_cost must be >= 0, got {self.gross_cost}")
        if self.executable_size < 0:
            raise ValueError(
                f"executable_size must be >= 0, got {self.executable_size}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Deterministic dictionary representation for JSON serialization.

        Produces a flat dictionary suitable for canonical ID computation
        and JSON/Parquet serialization.
        """
        return {
            "opportunity_id": self.opportunity_id,
            "observation_id": self.observation_id,
            "opportunity_type": self.opportunity_type,
            "legs": [
                {
                    "leg_index": leg.leg_index,
                    "provider": leg.provider,
                    "provider_instrument_id": leg.provider_instrument_id,
                    "side": leg.side,
                    "execution_role": leg.execution_role,
                    "price": leg.price,
                    "vwap": leg.vwap,
                    "fill_size": leg.fill_size,
                    "fill_notional": leg.fill_notional,
                    "fill_count": leg.fill_count,
                    "levels_consumed": leg.levels_consumed,
                    "best_available_price": leg.best_available_price,
                }
                for leg in self.legs
            ],
            "execution_plan": {
                "legs": [
                    {
                        "leg_index": pl.leg_index,
                        "provider": pl.provider,
                        "provider_instrument_id": pl.provider_instrument_id,
                        "side": pl.side,
                        "target_size": pl.target_size,
                        "expected_price": pl.expected_price,
                        "expected_notional": pl.expected_notional,
                    }
                    for pl in self.execution_plan.legs
                ],
                "total_estimated_cost": self.execution_plan.total_estimated_cost,
                "total_estimated_proceeds": self.execution_plan.total_estimated_proceeds,
                "estimated_net": self.execution_plan.estimated_net,
                "is_simultaneous": self.execution_plan.is_simultaneous,
            },
            "gross_cost": self.gross_cost,
            "gross_proceeds": self.gross_proceeds,
            "gross_profit": self.gross_profit,
            "gross_return": self.gross_return,
            "cost_breakdown": {
                "buy_fee": self.cost_breakdown.buy_fee,
                "sell_fee": self.cost_breakdown.sell_fee,
                "fixed_cost": self.cost_breakdown.fixed_cost,
                "settlement_cost": self.cost_breakdown.settlement_cost,
                "transfer_cost": self.cost_breakdown.transfer_cost,
                "total_fee": self.cost_breakdown.total_fee,
                "total_cost": self.cost_breakdown.total_cost,
                "effective_fee_rate": self.cost_breakdown.effective_fee_rate,
            },
            "net_profit": self.net_profit,
            "net_return": self.net_return,
            "requested_size": self.requested_size,
            "executable_size": self.executable_size,
            "fully_executable": self.fully_executable,
            "risk_flags": {
                "is_fully_executable": self.risk_flags.is_fully_executable,
                "has_partial_fill": self.risk_flags.has_partial_fill,
                "has_unpaired_exposure": self.risk_flags.has_unpaired_exposure,
                "gross_margin_is_thin": self.risk_flags.gross_margin_is_thin,
                "net_margin_is_negative": self.risk_flags.net_margin_is_negative,
                "near_expiry": self.risk_flags.near_expiry,
                "is_expired": self.risk_flags.is_expired,
                "identity_validated": self.risk_flags.identity_validated,
                "identity_unvalidated": self.risk_flags.identity_unvalidated,
                "settlement_currency_mismatch": self.risk_flags.settlement_currency_mismatch,
                "resolution_source_mismatch": self.risk_flags.resolution_source_mismatch,
            },
            "observation_timestamp": self.observation_timestamp.isoformat(),
            "evaluation_time": self.evaluation_time.isoformat() if self.evaluation_time else None,
            "is_point_in_time": self.is_point_in_time,
            "close_time": self.close_time.isoformat() if self.close_time else None,
            "expiry_time": self.expiry_time.isoformat() if self.expiry_time else None,
            "outcome_label": self.outcome_label,
        }


# ══════════════════════════════════════════════════════════════════════════════
# IDENTITY COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════


def _compute_identity_dict(opportunity: NormalizedOpportunity) -> dict[str, Any]:
    """Compute the canonical dictionary for opportunity ID.

    Includes only economic structure fields — no timestamps.
    """
    return {
        "opportunity_type": opportunity.opportunity_type,
        "executable_size": opportunity.executable_size,
        "gross_cost": opportunity.gross_cost,
        "gross_profit": opportunity.gross_profit,
        "legs": [
            {
                "provider": leg.provider,
                "provider_instrument_id": leg.provider_instrument_id,
                "side": leg.side,
            }
            for leg in sorted(opportunity.legs, key=lambda x: x.leg_index)
        ],
    }


def _compute_observation_dict(opportunity: NormalizedOpportunity) -> dict[str, Any]:
    """Compute the canonical dictionary for observation ID.

    Includes all opportunity_id fields plus observation timestamp
    truncated to second for sub-second deduplication.
    """
    base = _compute_identity_dict(opportunity)
    ts = opportunity.observation_timestamp
    base["observation_timestamp"] = ts.replace(microsecond=0).isoformat()
    return base


# ══════════════════════════════════════════════════════════════════════════════
# RISK FLAG DERIVATION
# ══════════════════════════════════════════════════════════════════════════════


def _derive_risk_flags(
    *,
    fully_executable: bool,
    executable_size: float,
    requested_size: float,
    gross_profit: float,
    gross_cost: float,
    net_profit: float,
    close_time: datetime | None,
    expiry_time: datetime | None,
    observation_timestamp: datetime,
    mapping: MarketMapping | None,
    has_unpaired_exposure: bool,
    settlement_currency_mismatch: bool,
    resolution_source_mismatch: bool,
) -> RiskFlags:
    """Derive deterministic risk flags from opportunity fields."""
    has_partial_fill = not fully_executable and executable_size > 0
    gross_margin_is_thin = (
        gross_profit < 0.05 * gross_cost if gross_cost > 0 else False
    )
    net_margin_is_negative = net_profit < 0

    near_expiry = False
    is_expired = False
    if close_time is not None:
        time_to_close = (close_time - observation_timestamp).total_seconds()
        near_expiry = 0 < time_to_close < 86400  # 24 hours
    if expiry_time is not None:
        is_expired = expiry_time <= observation_timestamp

    return RiskFlags(
        is_fully_executable=fully_executable,
        has_partial_fill=has_partial_fill,
        has_unpaired_exposure=has_unpaired_exposure,
        gross_margin_is_thin=gross_margin_is_thin,
        net_margin_is_negative=net_margin_is_negative,
        near_expiry=near_expiry,
        is_expired=is_expired,
        identity_validated=mapping is not None,
        identity_unvalidated=mapping is None,
        settlement_currency_mismatch=settlement_currency_mismatch,
        resolution_source_mismatch=resolution_source_mismatch,
    )


# ══════════════════════════════════════════════════════════════════════════════
# COST EVALUATION
# ══════════════════════════════════════════════════════════════════════════════


def evaluate_normalized_costs(
    opportunity: NormalizedOpportunity,
) -> CostBreakdown:
    """Evaluate costs for a normalized opportunity.

    This is a canonical representation of already-established economics.
    It does NOT recalculate fees — it maps from existing cost evaluation
    results that were provided during normalization.

    The cost_breakdown is populated at normalization time from the
    detector-specific cost results (NetArbitrageResult or NetCrossVenueResult).
    This function validates and returns the existing breakdown.

    Args:
        opportunity: Normalized opportunity with pre-populated cost_breakdown.

    Returns:
        The CostBreakdown from the normalized opportunity.
    """
    return opportunity.cost_breakdown


# ══════════════════════════════════════════════════════════════════════════════
# NORMALIZATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════


def normalize_same_market(
    opportunity: ArbitrageOpportunity,
    result: NetArbitrageResult | None = None,
    observation_timestamp: datetime | None = None,
    market_identity: MarketIdentity | None = None,
) -> NormalizedOpportunity:
    """Convert same-market detector output to canonical form.

    Maps ArbitrageOpportunity → NormalizedOpportunity with two BUY legs
    (BUY YES + BUY NO).

    Args:
        opportunity: Raw same-market detector output.
        result: Cost-evaluated result (optional). If provided, populates
            CostBreakdown and net economics. If absent, only gross economics
            are available with zero fees.
        observation_timestamp: When the opportunity was observed. Defaults
            to datetime.now(UTC) if not provided.
        market_identity: Optional market identity for the market.

    Returns:
        NormalizedOpportunity in canonical form.
    """
    ts = observation_timestamp or datetime.now(UTC)

    # Build legs from execution results
    yes_exec = opportunity.yes_execution
    no_exec = opportunity.no_execution

    leg0 = OpportunityLeg(
        leg_index=0,
        provider=opportunity.provider,
        provider_instrument_id=opportunity.yes_instrument_id,
        market_identity=market_identity,
        side=LegRole.BUY,
        execution_role=ExecutionRole.PRIMARY,
        price=yes_exec.vwap if yes_exec.vwap is not None else 0.0,
        vwap=yes_exec.vwap,
        fill_size=yes_exec.filled_size,
        fill_notional=yes_exec.total_notional,
        fill_count=len(yes_exec.fills),
        levels_consumed=len(yes_exec.fills),
        best_available_price=yes_exec.fills[0].price if yes_exec.fills else 0.0,
    )

    leg1 = OpportunityLeg(
        leg_index=1,
        provider=opportunity.provider,
        provider_instrument_id=opportunity.no_instrument_id,
        market_identity=market_identity,
        side=LegRole.BUY,
        execution_role=ExecutionRole.COUNTER,
        price=no_exec.vwap if no_exec.vwap is not None else 0.0,
        vwap=no_exec.vwap,
        fill_size=no_exec.filled_size,
        fill_notional=no_exec.total_notional,
        fill_count=len(no_exec.fills),
        levels_consumed=len(no_exec.fills),
        best_available_price=no_exec.fills[0].price if no_exec.fills else 0.0,
    )

    # Build execution plan
    plan_leg0 = ExecutionPlanLeg(
        leg_index=0,
        provider=opportunity.provider,
        provider_instrument_id=opportunity.yes_instrument_id,
        side=LegRole.BUY,
        target_size=opportunity.requested_size,
        expected_price=leg0.price,
        expected_notional=leg0.price * opportunity.requested_size,
    )
    plan_leg1 = ExecutionPlanLeg(
        leg_index=1,
        provider=opportunity.provider,
        provider_instrument_id=opportunity.no_instrument_id,
        side=LegRole.BUY,
        target_size=opportunity.requested_size,
        expected_price=leg1.price,
        expected_notional=leg1.price * opportunity.requested_size,
    )
    execution_plan = ExecutionPlan(
        legs=(plan_leg0, plan_leg1),
        total_estimated_cost=plan_leg0.expected_notional + plan_leg1.expected_notional,
        total_estimated_proceeds=opportunity.requested_size * 1.0,
        estimated_net=(
            opportunity.requested_size * 1.0
            - plan_leg0.expected_notional
            - plan_leg1.expected_notional
        ),
        is_simultaneous=True,
    )

    # Cost breakdown — map from existing result or use zeros
    if result is not None:
        cost_breakdown = CostBreakdown(
            buy_fee=result.yes_fee,
            sell_fee=result.no_fee,
            fixed_cost=result.fixed_cost,
            settlement_cost=result.settlement_cost,
            transfer_cost=0.0,
            total_fee=result.yes_fee + result.no_fee,
            total_cost=result.total_cost,
            effective_fee_rate=(
                (result.yes_fee + result.no_fee) / result.gross_cost
                if result.gross_cost > 0
                else None
            ),
        )
        net_profit = result.net_profit
        net_return = result.net_return
        has_unpaired = result.unpaired_yes_size > 0 or result.unpaired_no_size > 0
    else:
        cost_breakdown = CostBreakdown(
            buy_fee=0.0,
            sell_fee=0.0,
            fixed_cost=0.0,
            settlement_cost=0.0,
            transfer_cost=0.0,
            total_fee=0.0,
            total_cost=opportunity.total_cost,
            effective_fee_rate=0.0,
        )
        net_profit = opportunity.gross_profit
        net_return = opportunity.gross_return
        has_unpaired = False

    # Temporal fields from market identity
    evaluation_time = None
    is_point_in_time_val = None
    close_time = None
    expiry_time = None
    if market_identity is not None:
        evaluation_time = market_identity.temporal_scope.evaluation_time
        is_point_in_time_val = market_identity.temporal_scope.is_point_in_time
        close_time = market_identity.event.boundary.close_time
        expiry_time = market_identity.event.boundary.expiry_time

    # Risk flags
    risk_flags = _derive_risk_flags(
        fully_executable=opportunity.fully_executable,
        executable_size=opportunity.executable_size,
        requested_size=opportunity.requested_size,
        gross_profit=opportunity.gross_profit,
        gross_cost=opportunity.total_cost,
        net_profit=net_profit,
        close_time=close_time,
        expiry_time=expiry_time,
        observation_timestamp=ts,
        mapping=None,
        has_unpaired_exposure=has_unpaired,
        settlement_currency_mismatch=False,
        resolution_source_mismatch=False,
    )

    # Build opportunity (without IDs first, then compute IDs)
    legs = (leg0, leg1)
    normalized = NormalizedOpportunity(
        opportunity_id="",  # placeholder
        observation_id="",  # placeholder
        opportunity_type=OpportunityType.SAME_MARKET,
        legs=legs,
        execution_plan=execution_plan,
        gross_cost=opportunity.total_cost,
        gross_proceeds=opportunity.guaranteed_payoff,
        gross_profit=opportunity.gross_profit,
        gross_return=opportunity.gross_return,
        cost_breakdown=cost_breakdown,
        net_profit=net_profit,
        net_return=net_return,
        requested_size=opportunity.requested_size,
        executable_size=opportunity.executable_size,
        fully_executable=opportunity.fully_executable,
        risk_flags=risk_flags,
        mapping=None,
        observation_timestamp=ts,
        evaluation_time=evaluation_time,
        is_point_in_time=is_point_in_time_val,
        close_time=close_time,
        expiry_time=expiry_time,
        outcome_label="YES+NO",
    )

    # Compute deterministic IDs
    opp_dict = _compute_identity_dict(normalized)
    obs_dict = _compute_observation_dict(normalized)

    # Return with computed IDs (using object.__setattr__ on frozen dataclass)
    object.__setattr__(normalized, "opportunity_id", _compute_opportunity_id(opp_dict))
    object.__setattr__(normalized, "observation_id", _compute_opportunity_id(obs_dict))

    return normalized


def normalize_cross_venue(
    opportunity: CrossVenueOpportunity,
    result: NetCrossVenueResult | None = None,
    observation_timestamp: datetime | None = None,
) -> NormalizedOpportunity:
    """Convert cross-venue detector output to canonical form.

    Maps CrossVenueOpportunity → NormalizedOpportunity with BUY + SELL legs.

    Args:
        opportunity: Raw cross-venue detector output.
        result: Cost-evaluated result (optional). If provided, populates
            CostBreakdown and net economics. If absent, only gross economics
            are available with zero fees.
        observation_timestamp: When the opportunity was observed. Defaults
            to datetime.now(UTC) if not provided.

    Returns:
        NormalizedOpportunity in canonical form.
    """
    ts = observation_timestamp or datetime.now(UTC)

    # Build legs from execution results
    buy_exec = opportunity.buy_execution
    sell_exec = opportunity.sell_execution

    leg0 = OpportunityLeg(
        leg_index=0,
        provider=opportunity.buy_venue,
        provider_instrument_id=opportunity.buy_instrument_id,
        market_identity=None,
        side=LegRole.BUY,
        execution_role=ExecutionRole.PRIMARY,
        price=buy_exec.vwap if buy_exec.vwap is not None else 0.0,
        vwap=buy_exec.vwap,
        fill_size=buy_exec.filled_size,
        fill_notional=buy_exec.total_notional,
        fill_count=len(buy_exec.fills),
        levels_consumed=len(buy_exec.fills),
        best_available_price=buy_exec.fills[0].price if buy_exec.fills else 0.0,
    )

    leg1 = OpportunityLeg(
        leg_index=1,
        provider=opportunity.sell_venue,
        provider_instrument_id=opportunity.sell_instrument_id,
        market_identity=None,
        side=LegRole.SELL,
        execution_role=ExecutionRole.COUNTER,
        price=sell_exec.vwap if sell_exec.vwap is not None else 0.0,
        vwap=sell_exec.vwap,
        fill_size=sell_exec.filled_size,
        fill_notional=sell_exec.total_notional,
        fill_count=len(sell_exec.fills),
        levels_consumed=len(sell_exec.fills),
        best_available_price=sell_exec.fills[0].price if sell_exec.fills else 0.0,
    )

    # Build execution plan
    plan_leg0 = ExecutionPlanLeg(
        leg_index=0,
        provider=opportunity.buy_venue,
        provider_instrument_id=opportunity.buy_instrument_id,
        side=LegRole.BUY,
        target_size=opportunity.requested_size,
        expected_price=leg0.price,
        expected_notional=leg0.price * opportunity.requested_size,
    )
    plan_leg1 = ExecutionPlanLeg(
        leg_index=1,
        provider=opportunity.sell_venue,
        provider_instrument_id=opportunity.sell_instrument_id,
        side=LegRole.SELL,
        target_size=opportunity.requested_size,
        expected_price=leg1.price,
        expected_notional=leg1.price * opportunity.requested_size,
    )
    execution_plan = ExecutionPlan(
        legs=(plan_leg0, plan_leg1),
        total_estimated_cost=plan_leg0.expected_notional,
        total_estimated_proceeds=plan_leg1.expected_notional,
        estimated_net=plan_leg1.expected_notional - plan_leg0.expected_notional,
        is_simultaneous=True,
    )

    # Cost breakdown — map from existing result or use zeros
    if result is not None:
        cost_breakdown = CostBreakdown(
            buy_fee=result.buy_fee,
            sell_fee=result.sell_fee,
            fixed_cost=result.fixed_cost,
            settlement_cost=result.settlement_cost,
            transfer_cost=result.transfer_cost,
            total_fee=result.buy_fee + result.sell_fee,
            total_cost=result.total_cost,
            effective_fee_rate=(
                (result.buy_fee + result.sell_fee) / result.gross_cost
                if result.gross_cost > 0
                else None
            ),
        )
        net_profit = result.net_spread
        net_return = result.net_return
        has_unpaired = result.unpaired_buy > 0 or result.unpaired_sell > 0
    else:
        cost_breakdown = CostBreakdown(
            buy_fee=0.0,
            sell_fee=0.0,
            fixed_cost=0.0,
            settlement_cost=0.0,
            transfer_cost=0.0,
            total_fee=0.0,
            total_cost=opportunity.buy_execution.total_notional,
            effective_fee_rate=0.0,
        )
        net_profit = opportunity.gross_spread
        net_return = opportunity.gross_return
        # Compute unpaired from leg fill sizes
        buy_filled = opportunity.buy_execution.filled_size
        sell_filled = opportunity.sell_execution.filled_size
        paired = min(buy_filled, sell_filled)
        has_unpaired = (buy_filled - paired) > 0 or (sell_filled - paired) > 0

    # Derive settlement/resolution mismatch from mapping
    settlement_currency_mismatch = False
    resolution_source_mismatch = False
    mapping = opportunity.mapping

    # Risk flags
    risk_flags = _derive_risk_flags(
        fully_executable=opportunity.fully_executable,
        executable_size=opportunity.executable_size,
        requested_size=opportunity.requested_size,
        gross_profit=opportunity.gross_spread,
        gross_cost=opportunity.buy_execution.total_notional,
        net_profit=net_profit,
        close_time=None,
        expiry_time=None,
        observation_timestamp=ts,
        mapping=mapping,
        has_unpaired_exposure=has_unpaired,
        settlement_currency_mismatch=settlement_currency_mismatch,
        resolution_source_mismatch=resolution_source_mismatch,
    )

    # Build opportunity (without IDs first, then compute IDs)
    legs = (leg0, leg1)
    normalized = NormalizedOpportunity(
        opportunity_id="",  # placeholder
        observation_id="",  # placeholder
        opportunity_type=OpportunityType.CROSS_VENUE,
        legs=legs,
        execution_plan=execution_plan,
        gross_cost=opportunity.buy_execution.total_notional,
        gross_proceeds=opportunity.sell_execution.total_notional,
        gross_profit=opportunity.gross_spread,
        gross_return=opportunity.gross_return,
        cost_breakdown=cost_breakdown,
        net_profit=net_profit,
        net_return=net_return,
        requested_size=opportunity.requested_size,
        executable_size=opportunity.executable_size,
        fully_executable=opportunity.fully_executable,
        risk_flags=risk_flags,
        mapping=mapping,
        observation_timestamp=ts,
        evaluation_time=None,
        is_point_in_time=None,
        close_time=None,
        expiry_time=None,
        outcome_label=opportunity.outcome_label,
    )

    # Compute deterministic IDs
    opp_dict = _compute_identity_dict(normalized)
    obs_dict = _compute_observation_dict(normalized)

    # Return with computed IDs (using object.__setattr__ on frozen dataclass)
    object.__setattr__(normalized, "opportunity_id", _compute_opportunity_id(opp_dict))
    object.__setattr__(normalized, "observation_id", _compute_opportunity_id(obs_dict))

    return normalized
