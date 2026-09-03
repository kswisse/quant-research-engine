"""Domain types for backtesting.

These define the interface for strategy evaluation and overfitting detection.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from quant_engine.statistics.types import StrategyStatistics


class BacktestResult(BaseModel):
    """Result of running a single strategy on historical data.

    This captures what the strategy "achieved" — which may be
    overfit when many strategies are tested.

    Fields match StrategyStatistics to simplify the bridge between
    backtesting and statistical evaluation.
    """

    model_config = {"frozen": True}

    strategy_name: str
    strategy_parameters: dict[str, float | int | str | bool] = Field(
        default_factory=dict
    )
    strategy_index: int = 0
    sharpe_ratio: float
    n_observations: int
    skewness: float
    kurtosis: float
    total_return: float
    annualized_return: float | None = None
    max_drawdown: float | None = None
    is_selected: bool = False

    def to_statistics(self) -> StrategyStatistics:
        """Convert to StrategyStatistics for statistical evaluation."""
        return StrategyStatistics(
            sharpe_ratio=self.sharpe_ratio,
            skewness=self.skewness,
            kurtosis=self.kurtosis,
            number_of_observations=self.n_observations,
        )


class OverfittingDiagnosis(BaseModel):
    """Diagnosis of backtest overfitting risk.

    Compares naive Sharpe to deflated Sharpe and estimates
    the probability that the best strategy is a false discovery.
    """

    model_config = {"frozen": True}

    naive_sharpe: float
    deflated_sharpe: float
    probability_of_overfitting: float
    n_trials: int
    skewness: float
    kurtosis: float
    n_observations: int
    is_overfit: bool
