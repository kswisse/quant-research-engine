"""Arbitrage detection and cost evaluation for prediction markets.

Public API:
    ArbitrageOpportunity: Detected same-market arbitrage opportunity
    CrossVenueOpportunity: Detected cross-venue arbitrage opportunity
    ArbitrageCostModel: Cost model for arbitrage execution
    NetArbitrageResult: Net arbitrage result with cost and exposure details
    NetCrossVenueResult: Net cross-venue result with cost and exposure details
    detect_same_market_arbitrage: Detect YES+NO arbitrage in a binary market
    detect_cross_venue_arbitrage: Detect cross-venue arbitrage between two venues
    evaluate_arbitrage_costs: Evaluate same-market arbitrage under execution costs
    evaluate_cross_venue_costs: Evaluate cross-venue arbitrage under execution costs

    NormalizedOpportunity: Canonical normalized arbitrage opportunity
    OpportunityLeg: One leg of a normalized opportunity
    ExecutionPlan: Theoretical execution plan
    CostBreakdown: Normalized cost representation
    RiskFlags: Static risk/quality flags
    normalize_same_market: Convert same-market result to canonical form
    normalize_cross_venue: Convert cross-venue result to canonical form
"""

from quant_engine.arbitrage.costs import (
    ArbitrageCostModel,
    NetArbitrageResult,
    NetCrossVenueResult,
    evaluate_arbitrage_costs,
    evaluate_cross_venue_costs,
)
from quant_engine.arbitrage.cross_venue import (
    CrossVenueOpportunity,
    detect_cross_venue_arbitrage,
)
from quant_engine.arbitrage.detection import detect_same_market_arbitrage
from quant_engine.arbitrage.errors import (
    ArbitrageError,
    IncompatibleOpportunityTypeError,
    InvalidLegError,
    NormalizationError,
)
from quant_engine.arbitrage.models import ArbitrageOpportunity
from quant_engine.arbitrage.normalization import (
    CostBreakdown,
    ExecutionPlan,
    ExecutionPlanLeg,
    ExecutionRole,
    LegRole,
    NormalizedOpportunity,
    OpportunityLeg,
    OpportunityType,
    RiskFlags,
    normalize_cross_venue,
    normalize_same_market,
)
from quant_engine.market_identity.mapping import MarketMapping

__all__ = [
    "ArbitrageCostModel",
    "ArbitrageError",
    "ArbitrageOpportunity",
    "CostBreakdown",
    "CrossVenueOpportunity",
    "ExecutionPlan",
    "ExecutionPlanLeg",
    "ExecutionRole",
    "IncompatibleOpportunityTypeError",
    "InvalidLegError",
    "LegRole",
    "MarketMapping",
    "NetArbitrageResult",
    "NetCrossVenueResult",
    "NormalizedOpportunity",
    "NormalizationError",
    "OpportunityLeg",
    "OpportunityType",
    "RiskFlags",
    "detect_cross_venue_arbitrage",
    "detect_same_market_arbitrage",
    "evaluate_arbitrage_costs",
    "evaluate_cross_venue_costs",
    "normalize_cross_venue",
    "normalize_same_market",
]
