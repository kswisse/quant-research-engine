"""Descriptive statistics for return series.

All functions operate on NumPy arrays of decimal returns (0.01 = 1%).
Conventions are documented per function.

Units: returns are decimal. Moments are dimensionless.
"""

from __future__ import annotations

import numpy as np

from quant_engine.core.errors import InsufficientDataError


def mean(returns: np.ndarray) -> float:
    """Arithmetic mean of returns.

    Formula: r_bar = (1/T) * sum(r_t)

    Args:
        returns: 1-D array of decimal returns.

    Returns:
        Arithmetic mean.

    Raises:
        InsufficientDataError: If returns has zero elements.
    """
    if returns.ndim != 1:
        raise ValueError(f"Expected 1-D array, got {returns.ndim}-D")
    if len(returns) == 0:
        raise InsufficientDataError("Cannot compute mean of empty series")
    return float(np.mean(returns))


def std(returns: np.ndarray, ddof: int = 1) -> float:
    """Sample standard deviation of returns.

    Formula: sqrt( (1/(T-ddof)) * sum((r_t - r_bar)^2) )

    Convention: Uses T-1 denominator (Bessel's correction) by default,
    matching the sample standard deviation used in Sharpe ratio calculations.

    Args:
        returns: 1-D array of decimal returns.
        ddof: Delta degrees of freedom. Default 1 (sample std).

    Returns:
        Standard deviation.

    Raises:
        InsufficientDataError: If returns has fewer elements than ddof + 1.
        ValueError: If returns is not 1-D.
    """
    if returns.ndim != 1:
        raise ValueError(f"Expected 1-D array, got {returns.ndim}-D")
    if len(returns) <= ddof:
        raise InsufficientDataError(
            f"Need at least {ddof + 1} observations for std with ddof={ddof}, got {len(returns)}"
        )
    return float(np.std(returns, ddof=ddof))


def skewness(returns: np.ndarray) -> float:
    """Sample skewness of returns (Fisher's definition).

    Formula: g1 = (1/T) * sum(((r_t - r_bar) / s)^3)

    where s is the sample standard deviation (T-1 denominator).

    Convention: This is the adjusted Fisher-Pearson skewness coefficient,
    matching scipy.stats.skew with bias=False (equivalently, default for
    pandas and most financial literature).

    A normal distribution has skewness = 0.

    Args:
        returns: 1-D array of decimal returns. Must have at least 3 observations.

    Returns:
        Skewness coefficient.

    Raises:
        InsufficientDataError: If fewer than 3 observations.
        ValueError: If returns is not 1-D.
    """
    if returns.ndim != 1:
        raise ValueError(f"Expected 1-D array, got {returns.ndim}-D")
    if len(returns) < 3:
        raise InsufficientDataError(
            f"Skewness requires at least 3 observations, got {len(returns)}"
        )
    T = len(returns)  # noqa: N806 — mathematical notation for sample size
    r_bar = np.mean(returns)
    s = np.std(returns, ddof=1)
    if s == 0:
        return 0.0
    m3 = np.mean(((returns - r_bar) / s) ** 3)
    # Adjusted Fisher-Pearson: multiply by T / ((T-1)*(T-2))
    return float(m3 * T * T / ((T - 1) * (T - 2)))


def kurtosis(returns: np.ndarray) -> float:
    """Sample kurtosis of returns (regular kurtosis, NOT excess kurtosis).

    Formula: K = (1/T) * sum(((r_t - r_bar) / s)^4)

    where s is the sample standard deviation (T-1 denominator).

    Convention: This returns the regular (non-excess) kurtosis.
    A normal distribution has kurtosis = 3.0.
    Excess kurtosis = kurtosis - 3.0.

    The Bailey & López de Prado (2014) PSR/DSR formulas use regular kurtosis
    (denoted γ₄ in their notation). The excess kurtosis adjustment appears
    inside the formula as (γ₄ - 1)/4, where γ₄ - 1 is the excess kurtosis.

    For the adjusted (bias-corrected) kurtosis used in PSR, see
    `adjusted_kurtosis`.

    Args:
        returns: 1-D array of decimal returns. Must have at least 4 observations.

    Returns:
        Kurtosis (regular, not excess). Normal = 3.0.

    Raises:
        InsufficientDataError: If fewer than 4 observations.
        ValueError: If returns is not 1-D.
    """
    if returns.ndim != 1:
        raise ValueError(f"Expected 1-D array, got {returns.ndim}-D")
    if len(returns) < 4:
        raise InsufficientDataError(
            f"Kurtosis requires at least 4 observations, got {len(returns)}"
        )
    T = len(returns)  # noqa: N806 — mathematical notation for sample size
    r_bar = np.mean(returns)
    s = np.std(returns, ddof=1)
    if s == 0:
        return 3.0  # Normal kurtosis as fallback for constant series
    m4 = np.mean(((returns - r_bar) / s) ** 4)
    # Bias-corrected kurtosis (matches scipy.stats.kurtosis with fisher=False, bias=False)
    raw_kurt = m4
    adjustment = (T - 1) / ((T - 2) * (T - 3)) * ((T + 1) * raw_kurt - 3 * (T - 1))
    return float(raw_kurt + adjustment)


def excess_kurtosis(returns: np.ndarray) -> float:
    """Excess kurtosis of returns (= kurtosis - 3).

    A normal distribution has excess kurtosis = 0.

    Args:
        returns: 1-D array of decimal returns.

    Returns:
        Excess kurtosis.
    """
    return kurtosis(returns) - 3.0


def validate_returns(returns: np.ndarray) -> np.ndarray:
    """Validate and clean a return series.

    Checks:
    - 1-D array
    - All values are finite (no NaN, +inf, -inf)
    - At least 2 observations

    Args:
        returns: Input array.

    Returns:
        Validated 1-D float64 array.

    Raises:
        ValueError: If not 1-D, contains non-few values, or has < 2 observations.
    """
    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D array, got {arr.ndim}-D")
    if len(arr) < 2:
        raise ValueError(f"Need at least 2 observations, got {len(arr)}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Returns contain non-finite values (NaN, inf)")
    return arr
