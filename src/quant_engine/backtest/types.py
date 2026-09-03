"""Domain types for backtesting.

These define the interface for strategy evaluation and overfitting detection.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Strategy(BaseModel):
    """A trading strategy definition.

    In Phase 0, strategies are defined by a name and parameters.
    Actual strategy logic will be added in the implementation phase.
    """

    model_config = {"frozen": True}

    name: str
    parameters: dict[str, float | int | str | bool] = Field(default_factory=dict)
    index: int = 0  # position in the trial sequence


class BacktestResult(BaseModel):
    """Result of running a single strategy on historical data.

    This captures what the strategy "achieved" — which may be
    overfit when many strategies are tested.
    """

    model_config = {"frozen": True}

    strategy: Strategy
    sharpe_ratio: float
    n_observations: int
    skewness: float
    kurtosis: float
    is_selected: bool = False  # was this the best-performing strategy?


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
    is_overfit: bool  # deflated_sharpe < 0 or probability_of_overfitting > threshold
