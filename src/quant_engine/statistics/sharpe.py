"""Sharpe Ratio implementation.

Reference:
    Sharpe, W.F. (1994). "The Sharpe Ratio". Journal of Portfolio Management, 21(1), 49-58.

Conventions (explicitly chosen):
    - Returns are decimal arithmetic returns (0.01 = 1%).
    - Risk-free rate defaults to 0 (excess returns = raw returns).
    - Volatility uses sample standard deviation (T-1 denominator, Bessel's correction).
    - The base function works with non-annualized periodic returns.
    - Annualization requires an explicit periods_per_year parameter.
"""

from __future__ import annotations

import numpy as np

from quant_engine.core.errors import NumericalInstabilityError
from quant_engine.statistics.descriptive import mean, std, validate_returns


def sharpe_ratio(
    returns: np.ndarray,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int | None = None,
) -> float:
    """Compute the Sharpe Ratio of a return series.

    Formula (non-annualized):
        SR = mean(r - rf) / std(r - rf)

    where:
        r = periodic returns
        rf = periodic risk-free rate
        std uses T-1 denominator (sample std)

    If periods_per_year is provided, annualizes:
        SR_annual = SR_periodic * sqrt(periods_per_year)

    Args:
        returns: 1-D array of decimal returns.
        risk_free_rate: Periodic risk-free rate (default 0.0).
        periods_per_year: If provided, annualize the result.

    Returns:
        Sharpe Ratio (dimensionless for periodic, annualized if periods_per_year set).

    Raises:
        ValueError: If returns is invalid.
        NumericalInstabilityError: If volatility is zero or effectively zero.
    """
    r = validate_returns(returns)
    excess = r - risk_free_rate
    mu = mean(excess)
    sigma = std(excess, ddof=1)
    if sigma < 1e-15:
        raise NumericalInstabilityError(
            f"Volatility is effectively zero ({sigma:.2e}), "
            "Sharpe ratio is undefined for constant returns"
        )
    sr = mu / sigma
    if periods_per_year is not None:
        sr *= np.sqrt(periods_per_year)
    return float(sr)
