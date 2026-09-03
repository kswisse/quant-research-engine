"""Tests for domain type validation."""

import pytest
from pydantic import ValidationError

from quant_engine.backtest.types import OverfittingDiagnosis, Strategy
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


class TestStrategy:
    def test_valid_creation(self):
        s = Strategy(name="momentum", parameters={"lookback": 20})
        assert s.name == "momentum"
        assert s.index == 0

    def test_frozen(self):
        s = Strategy(name="test")
        with pytest.raises(ValidationError):
            s.name = "other"


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
