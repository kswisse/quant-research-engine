"""Tests for Deflated Sharpe Ratio."""

import numpy as np
import pytest
from scipy.stats import norm

from quant_engine.statistics.dsr import (
    DSRResult,
    dsr,
    dsr_from_stats,
    expected_max_sharpe,
)


class TestExpectedMaxSharpe:
    def test_single_trial(self):
        # With 1 trial, expected max is 0 (no selection)
        assert expected_max_sharpe(1, variance_sr=1.0) == 0.0

    def test_grows_with_trials(self):
        # More trials → higher expected max
        em10 = expected_max_sharpe(10)
        em100 = expected_max_sharpe(100)
        em1000 = expected_max_sharpe(1000)
        assert em10 < em100 < em1000

    def test_grows_with_variance(self):
        em_low = expected_max_sharpe(100, variance_sr=0.5)
        em_high = expected_max_sharpe(100, variance_sr=2.0)
        assert em_low < em_high

    def test_zero_variance(self):
        assert expected_max_sharpe(100, variance_sr=0.0) == 0.0

    def test_negative_trials_raises(self):
        with pytest.raises(ValueError):
            expected_max_sharpe(0)

    def test_known_values(self):
        # Bailey & López de Prado 2014 reference implementation
        # EMC = 0.5772156649
        emc = 0.5772156649015329
        N = 100  # noqa: N806 — mathematical notation
        V = 1.0  # noqa: N806 — mathematical notation
        z = (1 - emc) * norm.ppf(1 - 1.0 / N) + emc * norm.ppf(1 - 1.0 / (N * np.e))
        expected = np.sqrt(V) * z
        result = expected_max_sharpe(N, V)
        assert abs(result - expected) < 1e-10

    def test_approximation_accuracy(self):
        # The approximation should give reasonable values
        # For N=100, V=1: expected max ≈ 2.3-2.5 (from literature)
        em = expected_max_sharpe(100, 1.0)
        assert 2.0 < em < 3.0

    def test_large_trials(self):
        # For very large N, expected max grows slowly (logarithmically)
        em_1k = expected_max_sharpe(1000)
        em_10k = expected_max_sharpe(10000)
        # Should grow but not linearly
        ratio = em_10k / em_1k
        assert ratio < 2.0  # Much less than 10x


class TestDSR:
    def test_returns_dsr_result(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 252)
        result = dsr(returns, n_trials=100)
        assert isinstance(result, DSRResult)

    def test_probability_bounds(self):
        rng = np.random.default_rng(42)
        for _ in range(10):
            returns = rng.normal(0, 0.02, 252)
            result = dsr(returns, n_trials=100)
            assert 0.0 <= result.probability <= 1.0

    def test_more_trials_reduces_dsr(self):
        # More trials → higher benchmark → lower DSR for same returns
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 252)
        r_few = dsr(returns, n_trials=10)
        r_many = dsr(returns, n_trials=1000)
        assert r_many.probability < r_few.probability

    def test_high_sr_high_dsr(self):
        # Strong positive returns should survive deflation with few trials
        # and very low variance of SR estimates
        rng = np.random.default_rng(42)
        returns = rng.normal(0.005, 0.01, 1000)
        result = dsr(returns, n_trials=5, variance_sr=0.01)
        assert result.probability > 0.9

    def test_is_overfit_flag(self):
        # With many trials and mediocre returns, is_overfit should be True
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.02, 252)
        result = dsr(returns, n_trials=1000)
        assert result.is_overfit == (result.probability < 0.5)

    def test_benchmark_increases_with_trials(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, 252)
        r1 = dsr(returns, n_trials=10)
        r2 = dsr(returns, n_trials=100)
        assert r2.benchmark_sharpe > r1.benchmark_sharpe


class TestDSRFromStats:
    def test_matches_dsr(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 252)

        from quant_engine.statistics.descriptive import kurtosis, skewness
        from quant_engine.statistics.sharpe import sharpe_ratio

        sr = sharpe_ratio(returns)
        s3 = skewness(returns)
        s4 = kurtosis(returns)

        r1 = dsr(returns, n_trials=100)
        r2 = dsr_from_stats(sr, 252, s3, s4, n_trials=100)
        assert abs(r1.probability - r2.probability) < 1e-10
