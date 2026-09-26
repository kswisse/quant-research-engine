"""Tests for experiment results and runner."""

from __future__ import annotations

import numpy as np
import pytest

from quant_engine.experiments.config import ExperimentConfig
from quant_engine.experiments.results import ExperimentResult, PerformanceSummary
from quant_engine.experiments.runner import ExperimentRunner


def _make_config(
    seed: int = 42,
    n_strategies: int = 10,
    n_prices: int = 100,
) -> ExperimentConfig:
    """Create a deterministic test config with synthetic price data."""
    rng = np.random.default_rng(seed)
    # Random walk prices starting at 100
    returns = rng.normal(0.0005, 0.01, size=n_prices - 1)
    prices = [100.0]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    return ExperimentConfig(seed=seed, num_strategies=n_strategies, price_data=prices)


class TestExperimentRunner:
    """Test the experiment runner end-to-end."""

    def test_generates_correct_number(self) -> None:
        config = _make_config(n_strategies=10)
        runner = ExperimentRunner()
        result = runner.run(config)
        assert len(result.strategy_results) == 10
        assert result.performance_summary.n_strategies == 10

    def test_all_results_have_specs(self) -> None:
        config = _make_config(n_strategies=5)
        runner = ExperimentRunner()
        result = runner.run(config)
        for sr in result.strategy_results:
            assert sr.strategy_spec is not None
            assert sr.strategy_spec.family is not None

    def test_all_strategy_ids_unique(self) -> None:
        config = _make_config(n_strategies=20)
        runner = ExperimentRunner()
        result = runner.run(config)
        ids = [sr.strategy_spec.strategy_id for sr in result.strategy_results]
        assert len(set(ids)) == 20

    def test_experiment_id_matches_config(self) -> None:
        config = _make_config(n_strategies=5)
        runner = ExperimentRunner()
        result = runner.run(config)
        assert result.experiment_id == config.experiment_id

    def test_sharpe_ratios_accessible(self) -> None:
        config = _make_config(n_strategies=10)
        runner = ExperimentRunner()
        result = runner.run(config)
        sharpes = result.sharpe_ratios
        assert len(sharpes) == 10
        assert all(isinstance(s, float) for s in sharpes)


class TestReproducibility:
    """Same config → same results."""

    def test_same_config_same_results(self) -> None:
        config = _make_config(seed=42, n_strategies=10)
        runner = ExperimentRunner()
        r1 = runner.run(config)
        r2 = runner.run(config)

        assert r1.experiment_id == r2.experiment_id
        assert len(r1.strategy_results) == len(r2.strategy_results)

        for s1, s2 in zip(r1.strategy_results, r2.strategy_results):
            assert s1.strategy_spec.strategy_id == s2.strategy_spec.strategy_id
            assert s1.sharpe_ratio == pytest.approx(s2.sharpe_ratio)
            assert s1.total_return == pytest.approx(s2.total_return)


class TestDifferentSeeds:
    """Different seeds produce different populations."""

    def test_different_seeds_different_results(self) -> None:
        c1 = _make_config(seed=1, n_strategies=10)
        c2 = _make_config(seed=2, n_strategies=10)
        runner = ExperimentRunner()
        r1 = runner.run(c1)
        r2 = runner.run(c2)

        # Strategy IDs should differ
        ids1 = {sr.strategy_spec.strategy_id for sr in r1.strategy_results}
        ids2 = {sr.strategy_spec.strategy_id for sr in r2.strategy_results}
        assert ids1 != ids2


class TestPerformanceSummary:
    """Summary statistics are internally consistent."""

    def test_summary_consistency(self) -> None:
        config = _make_config(n_strategies=50)
        runner = ExperimentRunner()
        result = runner.run(config)
        summary = result.performance_summary

        assert summary.n_strategies == 50
        assert summary.sharpe_min <= summary.sharpe_median <= summary.sharpe_max
        assert summary.sharpe_q5 <= summary.sharpe_q25 <= summary.sharpe_q75 <= summary.sharpe_q95
        assert summary.return_min <= summary.return_mean <= summary.return_max

    def test_sharpe_std_non_negative(self) -> None:
        config = _make_config(n_strategies=10)
        runner = ExperimentRunner()
        result = runner.run(config)
        assert result.performance_summary.sharpe_std >= 0.0


class TestErrorHandling:
    """Invalid inputs produce clear errors."""

    def test_rejects_empty_prices(self) -> None:
        config = ExperimentConfig(seed=42, num_strategies=5, price_data=[])
        runner = ExperimentRunner()
        with pytest.raises(Exception):
            runner.run(config)

    def test_rejects_single_price(self) -> None:
        config = ExperimentConfig(seed=42, num_strategies=5, price_data=[100.0])
        runner = ExperimentRunner()
        with pytest.raises(Exception):
            runner.run(config)


class TestEndToEnd:
    """Full pipeline integration test."""

    def test_full_pipeline(self) -> None:
        config = _make_config(seed=42, n_strategies=10, n_prices=252)
        runner = ExperimentRunner()
        result = runner.run(config)

        # Check structure
        assert isinstance(result, ExperimentResult)
        assert result.config == config
        assert len(result.strategy_results) == 10
        assert isinstance(result.performance_summary, PerformanceSummary)

        # Check each strategy result
        for sr in result.strategy_results:
            assert hasattr(sr, "strategy_spec")
            assert hasattr(sr, "sharpe_ratio")
            assert hasattr(sr, "total_return")
            assert hasattr(sr, "n_observations")
            assert sr.n_observations > 0
            assert isinstance(sr.sharpe_ratio, float)

        # Check serializability
        d = result.config.to_dict()
        assert "experiment_id" in d
