"""Statistical methods: Sharpe ratio, PSR, DSR, and related validated metrics.

Public API:
    - sharpe_ratio: Compute the Sharpe Ratio of a return series.
    - psr: Probabilistic Sharpe Ratio (probability true SR > benchmark).
    - psr_from_stats: PSR from pre-computed statistics.
    - dsr: Deflated Sharpe Ratio (corrects for selection bias).
    - dsr_from_stats: DSR from pre-computed statistics.
    - expected_max_sharpe: Expected maximum SR under multiple testing.
    - sharpe_standard_error: Standard error of the SR estimator.
    - PSRResult: Result dataclass for PSR.
    - DSRResult: Result dataclass for DSR.
    - Descriptive statistics: mean, std, skewness, kurtosis, excess_kurtosis.
"""

from quant_engine.statistics.descriptive import (
    excess_kurtosis,
    kurtosis,
    mean,
    skewness,
    std,
    validate_returns,
)
from quant_engine.statistics.dsr import DSRResult, dsr, dsr_from_stats, expected_max_sharpe
from quant_engine.statistics.psr import PSRResult, psr, psr_from_stats, sharpe_standard_error
from quant_engine.statistics.sharpe import sharpe_ratio

__all__ = [
    "DSRResult",
    "PSRResult",
    "dsr",
    "dsr_from_stats",
    "excess_kurtosis",
    "expected_max_sharpe",
    "kurtosis",
    "mean",
    "psr",
    "psr_from_stats",
    "sharpe_ratio",
    "sharpe_standard_error",
    "skewness",
    "std",
    "validate_returns",
]
