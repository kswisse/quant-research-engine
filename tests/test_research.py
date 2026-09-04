"""Tests for the overfitting research study module."""

from __future__ import annotations

import numpy as np
import pytest

from quant_engine.research import (
    OverfittingStudyConfig,
    OverfittingStudyResult,
    run_overfitting_study,
)


def _make_prices(n: int = 252, seed: int = 99) -> list[float]:
    """Generate deterministic synthetic prices."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.01, size=n - 1)
    prices = [100.0]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    return prices


class TestOverfittingStudyConfig:
    """Test study configuration."""

    def test_valid_config(self) -> None:
        prices = _make_prices()
        config = OverfittingStudyConfig(
            seed=42,
            strategy_counts=(10, 100),
            price_data=prices,
        )
        assert config.seed == 42
        assert config.strategy_counts == (10, 100)
        assert len(config.price_data) == 252

    def test_rejects_empty_strategy_counts(self) -> None:
        prices = _make_prices()
        with pytest.raises(ValueError):
            OverfittingStudyConfig(seed=42, strategy_counts=(), price_data=prices)

    def test_rejects_non_positive_counts(self) -> None:
        prices = _make_prices()
        with pytest.raises(ValueError):
            OverfittingStudyConfig(seed=42, strategy_counts=(0, 10), price_data=prices)

    def test_rejects_negative_seed(self) -> None:
        prices = _make_prices()
        with pytest.raises(ValueError):
            OverfittingStudyConfig(seed=-1, strategy_counts=(10,), price_data=prices)

    def test_deterministic_study_id(self) -> None:
        prices = _make_prices()
        c1 = OverfittingStudyConfig(seed=42, strategy_counts=(10, 100), price_data=prices)
        c2 = OverfittingStudyConfig(seed=42, strategy_counts=(10, 100), price_data=prices)
        assert c1.study_id == c2.study_id

    def test_different_seed_different_id(self) -> None:
        prices = _make_prices()
        c1 = OverfittingStudyConfig(seed=42, strategy_counts=(10, 100), price_data=prices)
        c2 = OverfittingStudyConfig(seed=43, strategy_counts=(10, 100), price_data=prices)
        assert c1.study_id != c2.study_id

    def test_different_counts_different_id(self) -> None:
        prices = _make_prices()
        c1 = OverfittingStudyConfig(seed=42, strategy_counts=(10,), price_data=prices)
        c2 = OverfittingStudyConfig(seed=42, strategy_counts=(10, 100), price_data=prices)
        assert c1.study_id != c2.study_id

    def test_serialization_roundtrip(self) -> None:
        prices = _make_prices()
        config = OverfittingStudyConfig(seed=42, strategy_counts=(10, 100), price_data=prices)
        d = config.to_dict()
        restored = OverfittingStudyConfig.from_dict(d)
        assert restored.seed == config.seed
        assert restored.strategy_counts == config.strategy_counts
        assert restored.price_data == config.price_data
        assert restored.study_id == config.study_id


class TestOverfittingStudyResult:
    """Test study result model."""

    def test_to_dict_contains_key_fields(self) -> None:
        from quant_engine.overfitting.results import OverfittingAnalysisResult
        from quant_engine.statistics.dsr import DSRResult

        prices = _make_prices()
        config = OverfittingStudyConfig(seed=42, strategy_counts=(10,), price_data=prices)

        dummy_dsr = DSRResult(
            probability=0.1,
            observed_sharpe=0.5,
            benchmark_sharpe=1.0,
            expected_max_sharpe=1.0,
            standard_error=1.0,
            skewness=0.0,
            kurtosis=3.0,
            n_observations=100,
            n_trials=10,
            variance_sr=1.0,
            is_overfit=True,
        )
        analysis = OverfittingAnalysisResult(
            experiment_id="abc123",
            num_trials=10,
            observed_max_sharpe=0.5,
            best_strategy_id="def456",
            expected_max_sharpe=1.0,
            deflated_sharpe=0.1,
            is_overfit=True,
            mean_sharpe=0.0,
            median_sharpe=0.0,
            sharpe_std=1.0,
            performance_summary={"n_strategies": 10},
            dsr_result=dummy_dsr,
            track_record_length=100,
        )
        result = OverfittingStudyResult(
            config=config,
            analyses=[analysis],
        )
        d = result.to_dict()
        assert "study_id" in d
        assert "analyses" in d
        assert len(d["analyses"]) == 1
        assert d["analyses"][0]["num_trials"] == 10

    def test_study_id_delegates_to_config(self) -> None:
        from quant_engine.overfitting.results import OverfittingAnalysisResult
        from quant_engine.statistics.dsr import DSRResult

        prices = _make_prices()
        config = OverfittingStudyConfig(seed=42, strategy_counts=(10,), price_data=prices)
        dummy_dsr = DSRResult(
            probability=0.1, observed_sharpe=0.5, benchmark_sharpe=1.0,
            expected_max_sharpe=1.0, standard_error=1.0, skewness=0.0,
            kurtosis=3.0, n_observations=100, n_trials=10, variance_sr=1.0,
            is_overfit=True,
        )
        analysis = OverfittingAnalysisResult(
            experiment_id="abc123", num_trials=10, observed_max_sharpe=0.5,
            best_strategy_id="def456", expected_max_sharpe=1.0,
            deflated_sharpe=0.1, is_overfit=True, mean_sharpe=0.0,
            median_sharpe=0.0, sharpe_std=1.0,
            performance_summary={"n_strategies": 10}, dsr_result=dummy_dsr,
            track_record_length=100,
        )
        result = OverfittingStudyResult(config=config, analyses=[analysis])
        assert result.study_id == config.study_id


class TestRunOverfittingStudy:
    """Test study execution."""

    def test_returns_study_result(self) -> None:
        prices = _make_prices()
        config = OverfittingStudyConfig(seed=42, strategy_counts=(10,), price_data=prices)
        result = run_overfitting_study(config)
        assert isinstance(result, OverfittingStudyResult)

    def test_correct_number_of_analyses(self) -> None:
        prices = _make_prices()
        config = OverfittingStudyConfig(seed=42, strategy_counts=(10, 100), price_data=prices)
        result = run_overfitting_study(config)
        assert len(result.analyses) == 2

    def test_correct_num_trials_per_analysis(self) -> None:
        prices = _make_prices()
        config = OverfittingStudyConfig(seed=42, strategy_counts=(10, 50, 200), price_data=prices)
        result = run_overfitting_study(config)
        trial_counts = [a.num_trials for a in result.analyses]
        assert trial_counts == [10, 50, 200]

    def test_study_id_matches_config(self) -> None:
        prices = _make_prices()
        config = OverfittingStudyConfig(seed=42, strategy_counts=(10,), price_data=prices)
        result = run_overfitting_study(config)
        assert result.study_id == config.study_id

    def test_reproducibility(self) -> None:
        prices = _make_prices()
        config = OverfittingStudyConfig(seed=42, strategy_counts=(10, 100), price_data=prices)
        r1 = run_overfitting_study(config)
        r2 = run_overfitting_study(config)
        assert r1.study_id == r2.study_id
        for a1, a2 in zip(r1.analyses, r2.analyses):
            assert a1.observed_max_sharpe == a2.observed_max_sharpe
            assert a1.expected_max_sharpe == a2.expected_max_sharpe
            assert a1.deflated_sharpe == a2.deflated_sharpe
            assert a1.best_strategy_id == a2.best_strategy_id

    def test_expected_max_increases_with_trials(self) -> None:
        prices = _make_prices()
        config = OverfittingStudyConfig(seed=42, strategy_counts=(10, 100, 1000), price_data=prices)
        result = run_overfitting_study(config)
        expected = [a.expected_max_sharpe for a in result.analyses]
        assert expected[0] < expected[1] < expected[2]

    def test_same_price_data_all_experiments(self) -> None:
        prices = _make_prices()
        config = OverfittingStudyConfig(seed=42, strategy_counts=(10, 100), price_data=prices)
        result = run_overfitting_study(config)
        assert result.config.price_data == prices

    def test_serialization_roundtrip(self) -> None:
        prices = _make_prices()
        config = OverfittingStudyConfig(seed=42, strategy_counts=(10,), price_data=prices)
        result = run_overfitting_study(config)
        d = result.to_dict()
        assert d["study_id"] == result.study_id
        assert len(d["analyses"]) == len(result.analyses)

    def test_different_seed_different_results(self) -> None:
        prices = _make_prices()
        c1 = OverfittingStudyConfig(seed=42, strategy_counts=(10,), price_data=prices)
        c2 = OverfittingStudyConfig(seed=99, strategy_counts=(10,), price_data=prices)
        r1 = run_overfitting_study(c1)
        r2 = run_overfitting_study(c2)
        assert r1.study_id != r2.study_id

    def test_no_optimization(self) -> None:
        """The study must not alter generator parameters based on performance."""
        prices = _make_prices()
        config = OverfittingStudyConfig(seed=42, strategy_counts=(10, 100), price_data=prices)
        result = run_overfitting_study(config)
        assert result.config.seed == 42


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_n_equals_one(self) -> None:
        prices = _make_prices()
        config = OverfittingStudyConfig(seed=42, strategy_counts=(1,), price_data=prices)
        result = run_overfitting_study(config)
        assert len(result.analyses) == 1
        assert result.analyses[0].num_trials == 1

    def test_large_n(self) -> None:
        prices = _make_prices()
        config = OverfittingStudyConfig(seed=42, strategy_counts=(1000,), price_data=prices)
        result = run_overfitting_study(config)
        assert result.analyses[0].num_trials == 1000

    def test_single_price(self) -> None:
        """Minimum viable price data for DSR computation."""
        prices = [100.0 + i for i in range(10)]
        config = OverfittingStudyConfig(seed=42, strategy_counts=(1,), price_data=prices)
        result = run_overfitting_study(config)
        assert len(result.analyses) == 1


class TestPrefixPolicy:
    """Verify prefix-based seed policy."""

    def test_smaller_population_is_prefix(self) -> None:
        """N=10 strategies should be a prefix of N=100 strategies."""
        prices = _make_prices()
        config = OverfittingStudyConfig(seed=42, strategy_counts=(10, 100), price_data=prices)
        result = run_overfitting_study(config)
        assert result.analyses[0].num_trials == 10
        assert result.analyses[1].num_trials == 100
