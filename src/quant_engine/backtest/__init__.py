"""Backtest overfitting detection: strategy generation, evaluation, and deflated Sharpe ratio."""

from quant_engine.backtest.data import PriceData
from quant_engine.backtest.engine import BacktestOutput, run_backtest
from quant_engine.backtest.strategies import (
    AlwaysFlat,
    AlwaysShort,
    BuyAndHold,
    SMACrossover,
)
from quant_engine.backtest.strategy import Strategy
from quant_engine.backtest.types import BacktestResult, OverfittingDiagnosis

__all__ = [
    "AlwaysFlat",
    "AlwaysShort",
    "BacktestOutput",
    "BacktestResult",
    "BuyAndHold",
    "OverfittingDiagnosis",
    "PriceData",
    "SMACrossover",
    "Strategy",
    "run_backtest",
]
