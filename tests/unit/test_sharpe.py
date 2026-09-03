"""Tests for Sharpe Ratio."""

import numpy as np
import pytest

from quant_engine.core.errors import NumericalInstabilityError
from quant_engine.statistics.sharpe import sharpe_ratio


class TestSharpeRatio:
    def test_known_values(self):
        # Returns: [0.01, 0.02, -0.01, 0.03, -0.02]
        # mean = 0.006, std ≈ 0.01949, SR ≈ 0.3078
        returns = np.array([0.01, 0.02, -0.01, 0.03, -0.02])
        sr = sharpe_ratio(returns)
        expected = np.mean(returns) / np.std(returns, ddof=1)
        assert abs(sr - expected) < 1e-10

    def test_zero_mean(self):
        returns = np.array([0.01, -0.01, 0.01, -0.01])
        sr = sharpe_ratio(returns)
        assert abs(sr) < 1e-10

    def test_positive_mean(self):
        returns = np.array([0.01, 0.02, 0.03, 0.04])
        sr = sharpe_ratio(returns)
        assert sr > 0

    def test_negative_mean(self):
        returns = np.array([-0.01, -0.02, -0.03, -0.04])
        sr = sharpe_ratio(returns)
        assert sr < 0

    def test_constant_returns_raises(self):
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        with pytest.raises(NumericalInstabilityError):
            sharpe_ratio(returns)

    def test_annualization(self):
        returns = np.array([0.01, 0.02, -0.01, 0.03, -0.02])
        sr_daily = sharpe_ratio(returns)
        sr_annual = sharpe_ratio(returns, periods_per_year=252)
        assert abs(sr_annual - sr_daily * np.sqrt(252)) < 1e-10

    def test_risk_free_rate(self):
        returns = np.array([0.01, 0.02, 0.03, 0.04])
        rf = 0.005
        sr_with_rf = sharpe_ratio(returns, risk_free_rate=rf)
        sr_manual = np.mean(returns - rf) / np.std(returns - rf, ddof=1)
        assert abs(sr_with_rf - sr_manual) < 1e-10

    def test_two_observations(self):
        returns = np.array([0.01, -0.01])
        sr = sharpe_ratio(returns)
        # mean=0, so sr=0
        assert abs(sr) < 1e-10

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            sharpe_ratio(np.array([]))

    def test_nan_raises(self):
        with pytest.raises(ValueError):
            sharpe_ratio(np.array([0.01, np.nan, 0.02]))
