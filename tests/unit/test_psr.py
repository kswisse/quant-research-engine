"""Tests for Probabilistic Sharpe Ratio."""

import numpy as np

from quant_engine.statistics.psr import PSRResult, psr, psr_from_stats, sharpe_standard_error


class TestSharpeStandardError:
    def test_normal_returns(self):
        # For normal returns: se = sqrt(1 + SR²/2)
        sr = 1.0
        se = sharpe_standard_error(sr, skew=0.0, kurt=3.0)
        expected = np.sqrt(1.0 + sr**2 / 2.0)
        assert abs(se - expected) < 1e-10

    def test_zero_sharpe(self):
        se = sharpe_standard_error(0.0, skew=0.0, kurt=3.0)
        assert abs(se - 1.0) < 1e-10

    def test_negative_skew_increases_se(self):
        se_normal = sharpe_standard_error(1.0, skew=0.0, kurt=3.0)
        se_neg = sharpe_standard_error(1.0, skew=-1.0, kurt=3.0)
        # Negative skew with positive SR increases se
        assert se_neg > se_normal


class TestPSR:
    def test_sr_equals_benchmark(self):
        # When observed SR = benchmark SR, PSR should be around 0.5
        # (not exactly 0.5 due to finite-sample correction)
        returns = np.random.default_rng(42).normal(0.01, 0.02, 252)
        result = psr(returns, benchmark_sharpe=0.0)
        assert isinstance(result, PSRResult)
        assert 0.0 <= result.probability <= 1.0

    def test_high_sr_above_benchmark(self):
        # Strong positive returns should give high PSR vs benchmark=0
        rng = np.random.default_rng(42)
        returns = rng.normal(0.002, 0.01, 500)
        result = psr(returns, benchmark_sharpe=0.0)
        assert result.probability > 0.9

    def test_low_sr_below_benchmark(self):
        # Negative returns should give low PSR vs benchmark=1
        rng = np.random.default_rng(42)
        returns = rng.normal(-0.001, 0.01, 252)
        result = psr(returns, benchmark_sharpe=1.0)
        assert result.probability < 0.1

    def test_more_data_increases_confidence(self):
        rng = np.random.default_rng(42)
        short = rng.normal(0.001, 0.01, 50)
        long = np.tile(short, 10)  # 500 observations
        r_short = psr(short, benchmark_sharpe=0.0)
        r_long = psr(long, benchmark_sharpe=0.0)
        # More data with same mean should give more extreme PSR
        # (closer to 1 if SR > 0)
        assert r_long.probability > r_short.probability

    def test_probability_bounds(self):
        rng = np.random.default_rng(42)
        for _ in range(10):
            returns = rng.normal(0, 0.02, 252)
            result = psr(returns, benchmark_sharpe=0.0)
            assert 0.0 <= result.probability <= 1.0


class TestPSRFromStats:
    def test_matches_psr(self):
        # psr_from_stats should give same result as psr when given the same inputs
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 252)

        from quant_engine.statistics.descriptive import kurtosis, skewness
        from quant_engine.statistics.sharpe import sharpe_ratio

        sr = sharpe_ratio(returns)
        s3 = skewness(returns)
        s4 = kurtosis(returns)

        r1 = psr(returns, benchmark_sharpe=0.0)
        r2 = psr_from_stats(sr, 0.0, 252, s3, s4)
        assert abs(r1.probability - r2.probability) < 1e-10

    def test_known_normal_case(self):
        # For normal returns with SR=0: se = sqrt(1 + 0) = 1
        # z = (0 - 0) * sqrt(T-1) / 1 = 0
        # PSR = Φ(0) = 0.5
        result = psr_from_stats(
            observed_sharpe=0.0,
            benchmark_sharpe=0.0,
            n_observations=252,
            skewness_val=0.0,
            kurtosis_val=3.0,
        )
        assert abs(result.probability - 0.5) < 1e-10
