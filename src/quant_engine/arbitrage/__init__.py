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
from quant_engine.arbitrage.models import ArbitrageOpportunity
from quant_engine.market_identity.mapping import MarketMapping

__all__ = [
    "ArbitrageCostModel",
    "ArbitrageOpportunity",
    "CrossVenueOpportunity",
    "MarketMapping",
    "NetArbitrageResult",
    "NetCrossVenueResult",
    "detect_cross_venue_arbitrage",
    "detect_same_market_arbitrage",
    "evaluate_arbitrage_costs",
    "evaluate_cross_venue_costs",
]
