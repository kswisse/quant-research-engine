"""Arbitrage detection for prediction markets.

Public API:
    ArbitrageOpportunity: Detected same-market arbitrage opportunity
    detect_same_market_arbitrage: Detect YES+NO arbitrage in a binary market
"""

from quant_engine.arbitrage.detection import detect_same_market_arbitrage
from quant_engine.arbitrage.models import ArbitrageOpportunity

__all__ = [
    "ArbitrageOpportunity",
    "detect_same_market_arbitrage",
]
