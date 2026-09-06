"""Arbitrage detection and cost evaluation for prediction markets.

Public API:
    ArbitrageOpportunity: Detected same-market arbitrage opportunity
    ArbitrageCostModel: Cost model for arbitrage execution
    NetArbitrageResult: Net arbitrage result with cost and exposure details
    detect_same_market_arbitrage: Detect YES+NO arbitrage in a binary market
    evaluate_arbitrage_costs: Evaluate arbitrage under execution costs
"""

from quant_engine.arbitrage.costs import (
    ArbitrageCostModel,
    NetArbitrageResult,
    evaluate_arbitrage_costs,
)
from quant_engine.arbitrage.detection import detect_same_market_arbitrage
from quant_engine.arbitrage.models import ArbitrageOpportunity

__all__ = [
    "ArbitrageCostModel",
    "ArbitrageOpportunity",
    "NetArbitrageResult",
    "detect_same_market_arbitrage",
    "evaluate_arbitrage_costs",
]
