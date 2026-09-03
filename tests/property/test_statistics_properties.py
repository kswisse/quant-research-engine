"""Property-based tests for statistics invariants using Hypothesis."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_engine.statistics.descriptive import mean
from quant_engine.statistics.psr import sharpe_standard_error
from quant_engine.statistics.sharpe import sharpe_ratio


def finite_floats(min_value=-1e4, max_value=1e4):
    return st.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
    )


class TestMeanProperties:
    @given(st.lists(finite_floats(), min_size=2, max_size=100))
    @settings(max_examples=200)
    def test_mean_bounded_by_values(self, values):
        arr = np.array(values)
        m = mean(arr)
        assert m >= min(values) - 1e-10
        assert m <= max(values) + 1e-10

    @given(
        st.lists(finite_floats(min_value=-1e3, max_value=1e3), min_size=2, max_size=50),
        st.floats(min_value=-1e3, max_value=1e3),
    )
    @settings(max_examples=200)
    def test_mean_shift(self, values, c):
        arr = np.array(values)
        assert abs(mean(arr + c) - (mean(arr) + c)) < 1e-6


class TestSharpeProperties:
    @given(
        st.lists(
            st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
            min_size=3,
            max_size=100,
        ),
    )
    @settings(max_examples=200)
    def test_sr_sign_matches_mean(self, values):
        arr = np.array(values)
        # Skip constant returns (undefined Sharpe)
        if np.std(arr) < 1e-10:
            return
        sr = sharpe_ratio(arr)
        m = mean(arr)
        if abs(m) > 1e-10:
            assert (sr > 0) == (m > 0)

    @given(
        st.lists(
            st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
            min_size=3,
            max_size=100,
        ),
    )
    @settings(max_examples=200)
    def test_sr_sign_reversal(self, values):
        arr = np.array(values)
        # Skip constant returns
        if np.std(arr) < 1e-10:
            return
        sr_pos = sharpe_ratio(arr)
        sr_neg = sharpe_ratio(-arr)
        assert abs(sr_pos + sr_neg) < 1e-10


class TestPSRProperties:
    @given(
        st.floats(min_value=-1.0, max_value=1.0),
        st.floats(min_value=-1.0, max_value=1.0),
        st.floats(min_value=3.0, max_value=10.0),
    )
    @settings(max_examples=200)
    def test_se_positive_with_valid_inputs(self, sr, skew, kurt):
        # Moderate ranges ensure se_squared stays positive
        se = sharpe_standard_error(sr, skew, kurt)
        assert se >= 0
