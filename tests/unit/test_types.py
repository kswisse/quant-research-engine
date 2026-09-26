"""Tests for domain type validation."""

import pytest
from pydantic import ValidationError

from quant_engine.backtest.types import BacktestResult, OverfittingDiagnosis
from quant_engine.statistics.types import (
    ExperimentConfig,
    ReturnSeries,
    StrategyStatistics,
)


class TestReturnSeries:
    def test_valid_creation(self):
        rs = ReturnSeries(values=[0.01, -0.02, 0.03], label="test")
        assert rs.values == [0.01, -0.02, 0.03]
        assert rs.label == "test"

    def test_frozen(self):
        rs = ReturnSeries(values=[0.01])
        with pytest.raises(ValidationError):
            rs.values = [0.02]


class TestStrategyStatistics:
    def test_valid_creation(self):
        stats = StrategyStatistics(
            sharpe_ratio=1.5,
            skewness=0.0,
            kurtosis=3.0,
            number_of_observations=252,
        )
        assert stats.sharpe_ratio == 1.5
        assert stats.number_of_trials == 1

    def test_rejects_invalid_sharpe(self):
        with pytest.raises(ValidationError):
            StrategyStatistics(
                sharpe_ratio="not_a_number",
                skewness=0.0,
                kurtosis=3.0,
                number_of_observations=252,
            )


class TestExperimentConfig:
    def test_valid_creation(self):
        config = ExperimentConfig(name="test", seed=42, n_strategies=100)
        assert config.seed == 42
        assert config.significance_level == 0.05

    def test_rejects_zero_strategies(self):
        with pytest.raises(ValidationError):
            ExperimentConfig(name="test", seed=42, n_strategies=0)

    def test_rejects_invalid_significance(self):
        with pytest.raises(ValidationError):
            ExperimentConfig(name="test", seed=42, significance_level=0.0)


class TestBacktestResult:
    def test_valid_creation(self):
        result = BacktestResult(
            strategy_name="momentum",
            sharpe_ratio=1.5,
            n_observations=252,
            skewness=0.0,
            kurtosis=3.0,
            total_return=0.15,
        )
        assert result.strategy_name == "momentum"
        assert result.strategy_index == 0

    def test_frozen(self):
        result = BacktestResult(
            strategy_name="test",
            sharpe_ratio=1.0,
            n_observations=100,
            skewness=0.0,
            kurtosis=3.0,
            total_return=0.1,
        )
        with pytest.raises(ValidationError):
            result.strategy_name = "other"


class TestOverfittingDiagnosis:
    def test_valid_creation(self):
        diag = OverfittingDiagnosis(
            naive_sharpe=2.0,
            deflated_sharpe=0.5,
            probability_of_overfitting=0.3,
            n_trials=100,
            skewness=0.0,
            kurtosis=3.0,
            n_observations=252,
            is_overfit=False,
        )
        assert diag.is_overfit is False

    def test_is_overfit_flag(self):
        diag = OverfittingDiagnosis(
            naive_sharpe=3.0,
            deflated_sharpe=-0.2,
            probability_of_overfitting=0.9,
            n_trials=1000,
            skewness=0.0,
            kurtosis=3.0,
            n_observations=252,
            is_overfit=True,
        )
        assert diag.is_overfit is True
