"""Overfitting analysis result model.

Contains the complete output of a backtest overfitting analysis,
linking an experiment's results to multiple-testing-adjusted statistics.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from quant_engine.statistics.dsr import DSRResult  # noqa: TC001


class OverfittingAnalysisResult(BaseModel):
    """Result of a backtest overfitting analysis.

    This is a descriptive, probabilistic assessment — not a proof of
    overfitting or lack thereof. It quantifies how surprising the
    best observed Sharpe ratio is under a null model of zero skill,
    adjusted for the number of strategies tested.

    Attributes:
        experiment_id: Identifier linking to the source experiment.
        num_trials: Number of strategies tested (N).
        observed_max_sharpe: Best Sharpe ratio among the N strategies.
        best_strategy_id: ID of the strategy that achieved the max (observational only).
        expected_max_sharpe: Expected maximum Sharpe under the null of zero skill.
        deflated_sharpe: DSR probability that true SR > expected max.
        is_overfit: True if DSR < 0.5 (evidence of selection bias).
        mean_sharpe: Mean Sharpe across all strategies.
        median_sharpe: Median Sharpe across all strategies.
        sharpe_std: Standard deviation of Sharpe ratios.
        performance_summary: Reference to the experiment's PerformanceSummary.
        dsr_result: Full DSRResult from the statistical computation.
        independence_assumption: Documentation of the independent-trials assumption.
        variance_sr: Variance of SR estimates used in expected max computation.
        track_record_length: Number of return observations for the best strategy.
    """

    model_config = {"frozen": True}

    experiment_id: str
    num_trials: int = Field(ge=1)
    observed_max_sharpe: float
    best_strategy_id: str
    expected_max_sharpe: float
    deflated_sharpe: float
    is_overfit: bool
    mean_sharpe: float
    median_sharpe: float
    sharpe_std: float
    performance_summary: dict[str, object]
    dsr_result: DSRResult
    independence_assumption: str = (
        "Nominal trial count used. Generated strategies are NOT statistically "
        "independent (same-family strategies produce correlated signals). "
        "The effective number of independent trials may be smaller than N."
    )
    variance_sr: float = 1.0
    track_record_length: int = Field(ge=1)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "num_trials": self.num_trials,
            "observed_max_sharpe": self.observed_max_sharpe,
            "best_strategy_id": self.best_strategy_id,
            "expected_max_sharpe": self.expected_max_sharpe,
            "deflated_sharpe": self.deflated_sharpe,
            "is_overfit": self.is_overfit,
            "mean_sharpe": self.mean_sharpe,
            "median_sharpe": self.median_sharpe,
            "sharpe_std": self.sharpe_std,
            "variance_sr": self.variance_sr,
            "track_record_length": self.track_record_length,
            "independence_assumption": self.independence_assumption,
        }
