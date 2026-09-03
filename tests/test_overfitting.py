"""Tests for backtest overfitting analysis."""

from __future__ import annotations

import numpy as np
import pytest

from quant_engine.experiments.config import ExperimentConfig
from quant_engine.experiments.results import ExperimentResult, PerformanceSummary, StrategyResult
from quant_engine.experiments.runner import ExperimentRunner
from quant_engine.overfitting.analysis import analyze_backtest_overfitting
from quant_engine.overfitting.results import OverfittingAnalysisResult
from quant_engine.statistics.dsr import expected_max_sharpe


def _make_config(
    seed: int = 42,
    n_strategies: int = 10,
    n_prices: int = 252,
) -> ExperimentConfig:
    """Create a deterministic test config."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.01, size=n_prices - 1)
    prices = [100.0]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    return ExperimentConfig(seed=seed, num_strategies=n_strategies, price_data=prices)


class TestBasicAnalysis:
    """Test core analysis functionality."""

    def test_returns_analysis_result(self) -> None:
        config = _make_config(n_strategies=10)
        experiment = ExperimentRunner().run(config)
        result = analyze_backtest_overfitting(experiment)
        assert isinstance(result, OverfittingAnalysisResult)

    def test_correct_num_trials(self) -> None:
        config = _make_config(n_strategies=15)
        experiment = ExperimentRunner().run(config)
        result = analyze_backtest_overfitting(experiment)
        assert result.num_trials == 15

    def test_observed_max_is_maximum(self) -> None:
        config = _make_config(n_strategies=20)
        experiment = ExperimentRunner().run(config)
        result = analyze_backtest_overfitting(experiment)
        all_sharpes = [r.sharpe_ratio for r in experiment.strategy_results]
        assert result.observed_max_sharpe == max(all_sharpes)

    def test_best_strategy_id_matches(self) -> None:
        config = _make_config(n_strategies=10)
        experiment = ExperimentRunner().run(config)
        result = analyze_backtest_overfitting(experiment)
        # Find the strategy with the max Sharpe
        best = max(experiment.strategy_results, key=lambda r: r.sharpe_ratio)
        assert result.best_strategy_id == best.strategy_spec.strategy_id

    def test_experiment_id_matches(self) -> None:
        config = _make_config(n_strategies=10)
        experiment = ExperimentRunner().run(config)
        result = analyze_backtest_overfitting(experiment)
        assert result.experiment_id == experiment.experiment_id

    def test_mean_median_std_linkage(self) -> None:
        config = _make_config(n_strategies=20)
        experiment = ExperimentRunner().run(config)
        result = analyze_backtest_overfitting(experiment)
        sharpes = [r.sharpe_ratio for r in experiment.strategy_results]
        assert result.mean_sharpe == pytest.approx(np.mean(sharpes))
        assert result.median_sharpe == pytest.approx(np.median(sharpes))
        assert result.sharpe_std == pytest.approx(np.std(sharpes, ddof=1))


class TestExpectedMaximum:
    """Verify expected maximum comes from existing implementation."""

    def test_uses_existing_expected_max_sharpe(self) -> None:
        config = _make_config(n_strategies=50)
        experiment = ExperimentRunner().run(config)
        result = analyze_backtest_overfitting(experiment)
        expected = expected_max_sharpe(50, variance_sr=1.0)
        assert result.expected_max_sharpe == pytest.approx(expected)

    def test_expected_max_increases_with_trials(self) -> None:
        """More trials → larger expected maximum under the model."""
        configs = [_make_config(seed=42, n_strategies=n) for n in [10, 50, 200]]
        experiments = [ExperimentRunner().run(c) for c in configs]
        expecteds = [
            analyze_backtest_overfitting(e).expected_max_sharpe for e in experiments
        ]
        # Expected max should be non-decreasing with more trials
        assert expecteds[0] < expecteds[1] < expecteds[2]


class TestDSRIntegration:
    """Verify DSR uses existing implementation."""

    def test_dsr_is_probabilistic(self) -> None:
        config = _make_config(n_strategies=10)
        experiment = ExperimentRunner().run(config)
        result = analyze_backtest_overfitting(experiment)
        assert 0.0 <= result.deflated_sharpe <= 1.0

    def test_is_overfit_matches_dsr_threshold(self) -> None:
        config = _make_config(n_strategies=10)
        experiment = ExperimentRunner().run(config)
        result = analyze_backtest_overfitting(experiment)
        assert result.is_overfit == (result.deflated_sharpe < 0.5)

    def test_dsr_result_stored(self) -> None:
        config = _make_config(n_strategies=10)
        experiment = ExperimentRunner().run(config)
        result = analyze_backtest_overfitting(experiment)
        assert result.dsr_result.observed_sharpe == result.observed_max_sharpe
        assert result.dsr_result.n_trials == result.num_trials


class TestReproducibility:
    """Running analysis twice produces identical results."""

    def test_same_experiment_same_analysis(self) -> None:
        config = _make_config(n_strategies=10)
        experiment = ExperimentRunner().run(config)
        r1 = analyze_backtest_overfitting(experiment)
        r2 = analyze_backtest_overfitting(experiment)
        assert r1.observed_max_sharpe == r2.observed_max_sharpe
        assert r1.expected_max_sharpe == r2.expected_max_sharpe
        assert r1.deflated_sharpe == r2.deflated_sharpe
        assert r1.best_strategy_id == r2.best_strategy_id


class TestIndependenceAssumption:
    """Verify independence assumption is documented."""

    def test_assumption_documented(self) -> None:
        config = _make_config(n_strategies=10)
        experiment = ExperimentRunner().run(config)
        result = analyze_backtest_overfitting(experiment)
        assert "independent" in result.independence_assumption.lower()
        assert "correlated" in result.independence_assumption.lower()


class TestSerialization:
    """Test to_dict roundtrip."""

    def test_to_dict_contains_key_fields(self) -> None:
        config = _make_config(n_strategies=10)
        experiment = ExperimentRunner().run(config)
        result = analyze_backtest_overfitting(experiment)
        d = result.to_dict()
        assert "experiment_id" in d
        assert "num_trials" in d
        assert "observed_max_sharpe" in d
        assert "expected_max_sharpe" in d
        assert "deflated_sharpe" in d
        assert "best_strategy_id" in d
        assert "independence_assumption" in d

    def test_to_dict_deterministic(self) -> None:
        config = _make_config(n_strategies=10)
        experiment = ExperimentRunner().run(config)
        result = analyze_backtest_overfitting(experiment)
        d1 = result.to_dict()
        d2 = result.to_dict()
        assert d1 == d2


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_rejects_zero_strategies(self) -> None:
        # Manually construct an empty experiment result
        config = ExperimentConfig(seed=42, num_strategies=1, price_data=[100.0, 101.0])
        empty_result = ExperimentResult(
            config=config,
            strategy_results=[],
            performance_summary=PerformanceSummary(
                n_strategies=0,
                sharpe_min=0.0,
                sharpe_max=0.0,
                sharpe_mean=0.0,
                sharpe_median=0.0,
                sharpe_std=0.0,
                sharpe_q5=0.0,
                sharpe_q25=0.0,
                sharpe_q75=0.0,
                sharpe_q95=0.0,
                return_min=0.0,
                return_max=0.0,
                return_mean=0.0,
            ),
        )
        with pytest.raises(ValueError, match="zero strategy"):
            analyze_backtest_overfitting(empty_result)

    def test_n_equals_one(self) -> None:
        config = _make_config(n_strategies=1)
        experiment = ExperimentRunner().run(config)
        result = analyze_backtest_overfitting(experiment)
        assert result.num_trials == 1
        # With 1 trial, expected max should be 0 (from expected_max_sharpe)
        assert result.expected_max_sharpe == 0.0

    def test_rejects_nan_sharpe(self) -> None:
        config = _make_config(n_strategies=5)
        experiment = ExperimentRunner().run(config)
        # Inject NaN Sharpe
        corrupted = ExperimentResult(
            config=config,
            strategy_results=[
                StrategyResult(
                    strategy_spec=experiment.strategy_results[0].strategy_spec,
                    sharpe_ratio=float("nan"),
                    total_return=0.0,
                    n_observations=100,
                    skewness=0.0,
                    kurtosis=3.0,
                    mean_return=0.0,
                    volatility=0.01,
                )
            ],
            performance_summary=experiment.performance_summary,
        )
        with pytest.raises(ValueError, match="non-finite"):
            analyze_backtest_overfitting(corrupted)


class TestMoreTrials:
    """Demonstrate the central overfitting phenomenon."""

    def test_larger_n_increases_opportunity(self) -> None:
        """More strategies tested → larger expected maximum even with no skill."""
        small = analyze_backtest_overfitting(
            ExperimentRunner().run(_make_config(seed=42, n_strategies=10))
        )
        large = analyze_backtest_overfitting(
            ExperimentRunner().run(_make_config(seed=42, n_strategies=100))
        )
        # The expected maximum must be larger for more trials
        assert large.expected_max_sharpe > small.expected_max_sharpe
