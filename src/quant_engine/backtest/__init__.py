"""Backtest overfitting detection: strategy generation, evaluation, and deflated Sharpe ratio."""

from quant_engine.backtest.data import PriceData
from quant_engine.backtest.engine import BacktestOutput, run_backtest
from quant_engine.backtest.generator import (
    ALL_FAMILIES,
    RandomMeanReversionFamily,
    RandomMomentumFamily,
    RandomSMAFamily,
    RandomThresholdFamily,
    StrategyFamily,
    StrategyGenerator,
)
from quant_engine.backtest.strategies import (
    AlwaysFlat,
    AlwaysShort,
    BuyAndHold,
    SMACrossover,
)
from quant_engine.backtest.strategy import Strategy
from quant_engine.backtest.strategy_specs import ParameterSpace, StrategySpec
from quant_engine.backtest.types import BacktestResult, OverfittingDiagnosis

__all__ = [
    "ALL_FAMILIES",
    "AlwaysFlat",
    "AlwaysShort",
    "BacktestOutput",
    "BacktestResult",
    "BuyAndHold",
    "OverfittingDiagnosis",
    "ParameterSpace",
    "PriceData",
    "RandomMeanReversionFamily",
    "RandomMomentumFamily",
    "RandomSMAFamily",
    "RandomThresholdFamily",
    "SMACrossover",
    "Strategy",
    "StrategyFamily",
    "StrategyGenerator",
    "StrategySpec",
    "run_backtest",
]
