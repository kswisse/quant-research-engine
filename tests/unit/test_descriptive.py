"""Tests for descriptive statistics."""

import numpy as np
import pytest

from quant_engine.core.errors import InsufficientDataError
from quant_engine.statistics.descriptive import (
    excess_kurtosis,
    kurtosis,
    mean,
    skewness,
    std,
    validate_returns,
)


class TestMean:
    def test_simple(self):
        assert mean(np.array([1.0, 2.0, 3.0])) == 2.0

    def test_single_value(self):
        assert mean(np.array([5.0])) == 5.0

    def test_negative_values(self):
        assert mean(np.array([-1.0, 1.0])) == 0.0

    def test_empty_raises(self):
        with pytest.raises(InsufficientDataError):
            mean(np.array([]))

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            mean(np.array([[1.0, 2.0]]))


class TestStd:
    def test_known_values(self):
        data = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        result = std(data, ddof=1)
        expected = np.std(data, ddof=1)
        assert abs(result - expected) < 1e-12

    def test_single_value_raises(self):
        with pytest.raises(InsufficientDataError):
            std(np.array([1.0]), ddof=1)

    def test_two_values(self):
        result = std(np.array([1.0, 3.0]), ddof=1)
        assert abs(result - np.sqrt(2.0)) < 1e-12


class TestSkewness:
    def test_normal_approx(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 10000)
        s = skewness(data)
        assert abs(s) < 0.1  # Should be close to 0 for normal

    def test_symmetric(self):
        data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        s = skewness(data)
        assert abs(s) < 1e-10

    def test_too_few_raises(self):
        with pytest.raises(InsufficientDataError):
            skewness(np.array([1.0, 2.0]))

    def test_constant_returns_zero(self):
        data = np.array([1.0, 1.0, 1.0, 1.0])
        assert skewness(data) == 0.0


class TestKurtosis:
    def test_normal_approx(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 10000)
        k = kurtosis(data)
        assert abs(k - 3.0) < 0.2  # Should be close to 3 for normal

    def test_too_few_raises(self):
        with pytest.raises(InsufficientDataError):
            kurtosis(np.array([1.0, 2.0, 3.0]))

    def test_constant_returns_normal(self):
        data = np.array([1.0, 1.0, 1.0, 1.0])
        assert kurtosis(data) == 3.0


class TestExcessKurtosis:
    def test_normal_approx(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 10000)
        ek = excess_kurtosis(data)
        assert abs(ek) < 0.2  # Should be close to 0 for normal


class TestValidateReturns:
    def test_valid(self):
        data = np.array([0.01, -0.02, 0.03])
        result = validate_returns(data)
        assert result.dtype == np.float64

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="non-finite"):
            validate_returns(np.array([0.01, np.nan, 0.03]))

    def test_inf_raises(self):
        with pytest.raises(ValueError, match="non-finite"):
            validate_returns(np.array([0.01, np.inf, 0.03]))

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            validate_returns(np.array([0.01]))
