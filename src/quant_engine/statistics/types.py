"""Domain types for statistics and backtesting.

These types define the contracts between quantitative modules.
They use Pydantic for validation and immutability.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ReturnSeries(BaseModel):
    """A time series of returns.

    Represents period-over-period returns (not prices).
    All returns are decimal (0.01 = 1%).
    """

    model_config = {"frozen": True}

    label: str = ""
    values: list[float]
    frequency: str = "unknown"  # e.g., "daily", "hourly", "tick"


class StrategyStatistics(BaseModel):
    """Computed statistics for a single strategy or backtest.

    These are the raw inputs for overfitting analysis.
    """

    model_config = {"frozen": True}

    sharpe_ratio: float
    sortino_ratio: float | None = None
    max_drawdown: float | None = None
    skewness: float
    kurtosis: float
    number_of_observations: int
    number_of_trials: int = 1  # how many strategies were tested to find this one
    turnover: float | None = None
    annual_return: float | None = None
    annual_volatility: float | None = None


class ExperimentConfig(BaseModel):
    """Configuration for a single reproducible experiment.

    Captures all parameters needed to exactly reproduce a result.
    """

    model_config = {"frozen": True}

    name: str
    seed: int
    n_strategies: int = Field(ge=1, default=100)
    n_observations: int = Field(ge=2, default=252)
    significance_level: float = Field(gt=0, lt=1, default=0.05)
    parameters: dict[str, float | int | str | bool] = Field(default_factory=dict)


class ExperimentResult(BaseModel):
    """Results from a single experiment run.

    Contains both the naive and adjusted statistics for comparison.
    """

    model_config = {"frozen": True}

    config: ExperimentConfig
    naive_sharpe: float
    deflated_sharpe: float
    probability_of_overfitting: float
    best_strategy_index: int
    all_sharpe_ratios: list[float]
