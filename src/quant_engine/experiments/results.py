"""Experiment result models.

Contains the complete output of an experiment run, including
per-strategy results and summary statistics.
"""

from __future__ import annotations

from pydantic import BaseModel

from quant_engine.backtest.strategy_specs import StrategySpec  # noqa: TC001
from quant_engine.experiments.config import ExperimentConfig  # noqa: TC001


class StrategyResult(BaseModel):
    """Result for a single strategy within an experiment.

    Links a StrategySpec to its backtest performance.
    """

    model_config = {"frozen": True}

    strategy_spec: StrategySpec
    sharpe_ratio: float
    total_return: float
    n_observations: int
    skewness: float
    kurtosis: float
    mean_return: float
    volatility: float


class PerformanceSummary(BaseModel):
    """Aggregate performance distribution across all strategies.

    Descriptive statistics for the population of strategy results.
    Not a statistical significance claim.
    """

    model_config = {"frozen": True}

    n_strategies: int
    sharpe_min: float
    sharpe_max: float
    sharpe_mean: float
    sharpe_median: float
    sharpe_std: float
    sharpe_q5: float
    sharpe_q25: float
    sharpe_q75: float
    sharpe_q95: float
    return_min: float
    return_max: float
    return_mean: float


class ExperimentResult(BaseModel):
    """Complete result of an experiment run.

    Contains the configuration, per-strategy results, and summary.
    A researcher can trace from experiment → strategy → performance.
    """

    model_config = {"frozen": True}

    config: ExperimentConfig
    strategy_results: list[StrategyResult]
    performance_summary: PerformanceSummary

    @property
    def experiment_id(self) -> str:
        """Delegated to config.experiment_id."""
        return self.config.experiment_id

    @property
    def strategy_specs(self) -> list[StrategySpec]:
        """All strategy specifications."""
        return [r.strategy_spec for r in self.strategy_results]

    @property
    def sharpe_ratios(self) -> list[float]:
        """All Sharpe ratios from the experiment."""
        return [r.sharpe_ratio for r in self.strategy_results]
