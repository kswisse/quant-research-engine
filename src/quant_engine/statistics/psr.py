"""Probabilistic Sharpe Ratio (PSR).

Reference:
    Bailey, D.H. & López de Prado, M. (2012).
    "The Sharpe Ratio Efficient Frontier".
    Journal of Risk, 15(2), 3-44.

Conventions (explicitly chosen):
    - kurtosis parameter is REGULAR kurtosis (normal = 3.0), not excess kurtosis.
      The formula internally computes (kurtosis - 1)/4 as the excess adjustment.
    - The PSR uses the finite-sample standard error of the Sharpe estimator,
      accounting for skewness and kurtosis of the return distribution.
    - We use the Lo (2002) / Mertens (2002) correction for non-normal returns.

Mathematical formula:
    PSR(SR*) = Φ( (SR_hat - SR*) * sqrt(T-1) / se )

    where:
        se = sqrt( 1 - γ₃ * SR_hat + (γ₄ - 1)/4 * SR_hat² )

    and:
        Φ = standard normal CDF
        SR_hat = observed Sharpe ratio
        SR* = benchmark Sharpe ratio
        T = number of observations
        γ₃ = skewness of returns
        γ₄ = kurtosis of returns (regular, normal = 3.0)

    Note: (γ₄ - 1)/4 is the excess kurtosis adjustment. When γ₄ = 3 (normal),
    (γ₄ - 1)/4 = 2/4 = 0.5. When γ₄ = 1 (impossible for real returns),
    the term vanishes.

    Actually, for a normal distribution: se = sqrt(1 - 0 + (3-1)/4 * SR²) = sqrt(1 + SR²/2).
    This matches the standard result for the standard error of the Sharpe ratio
    under normality.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from quant_engine.core.errors import NumericalInstabilityError
from quant_engine.statistics.descriptive import kurtosis as compute_kurtosis
from quant_engine.statistics.descriptive import skewness, validate_returns
from quant_engine.statistics.sharpe import sharpe_ratio


@dataclass(frozen=True)
class PSRResult:
    """Result of a Probabilistic Sharpe Ratio computation.

    Attributes:
        probability: P(true SR > benchmark SR), in [0, 1].
        observed_sharpe: The estimated Sharpe ratio from the sample.
        benchmark_sharpe: The threshold Sharpe ratio being tested against.
        standard_error: The Lo (2002) standard error of the Sharpe estimator.
        skewness: Skewness of the return series.
        kurtosis: Regular kurtosis of the return series (normal = 3.0).
        n_observations: Number of return observations.
    """

    probability: float
    observed_sharpe: float
    benchmark_sharpe: float
    standard_error: float
    skewness: float
    kurtosis: float
    n_observations: int


def sharpe_standard_error(
    sr: float,
    skew: float,
    kurt: float,
) -> float:
    """Standard error of the Sharpe ratio estimator under non-normality.

    Formula (Lo 2002, Bailey & López de Prado 2012):
        se = sqrt( 1 - γ₃ * SR + (γ₄ - 1)/4 * SR² )

    where:
        SR = observed Sharpe ratio
        γ₃ = skewness of returns
        γ₄ = regular kurtosis (normal = 3.0)

    For normal returns (γ₃ = 0, γ₄ = 3):
        se = sqrt(1 + SR²/2)

    Args:
        sr: Observed Sharpe ratio.
        skew: Skewness of returns.
        kurt: Regular kurtosis of returns (normal = 3.0).

    Returns:
        Standard error (always positive for valid inputs).

    Raises:
        NumericalInstabilityError: If the argument under the square root is negative.
    """
    se_squared = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if se_squared < 0:
        raise NumericalInstabilityError(
            f"Standard error squared is negative ({se_squared:.6f}). "
            f"Inputs: sr={sr:.4f}, skew={skew:.4f}, kurt={kurt:.4f}. "
            "This indicates extreme skewness/kurtosis or a very large Sharpe ratio."
        )
    return float(np.sqrt(se_squared))


def psr(
    returns: np.ndarray,
    benchmark_sharpe: float = 0.0,
) -> PSRResult:
    """Compute the Probabilistic Sharpe Ratio.

    Args:
        returns: 1-D array of decimal returns.
        benchmark_sharpe: The threshold to test against (default 0.0).

    Returns:
        PSRResult with the probability that the true Sharpe > benchmark_sharpe.

    Raises:
        ValueError: If returns is invalid.
        NumericalInstabilityError: If computation is numerically unstable.
    """
    r = validate_returns(returns)
    T = len(r)  # noqa: N806 — mathematical notation for sample size
    sr_hat = sharpe_ratio(r)
    s3 = skewness(r)
    s4 = compute_kurtosis(r)  # Regular kurtosis, normal = 3.0
    se = sharpe_standard_error(sr_hat, s3, s4)
    if se < 1e-15:
        raise NumericalInstabilityError("Standard error is effectively zero, PSR is undefined")
    z = (sr_hat - benchmark_sharpe) * np.sqrt(T - 1) / se
    prob = float(norm.cdf(z))
    return PSRResult(
        probability=prob,
        observed_sharpe=sr_hat,
        benchmark_sharpe=benchmark_sharpe,
        standard_error=se,
        skewness=s3,
        kurtosis=s4,
        n_observations=T,
    )


def psr_from_stats(
    observed_sharpe: float,
    benchmark_sharpe: float,
    n_observations: int,
    skewness_val: float,
    kurtosis_val: float,
) -> PSRResult:
    """Compute PSR from pre-computed statistics (lower-level interface).

    Useful for testing or when statistics are already available.

    Args:
        observed_sharpe: Estimated Sharpe ratio.
        benchmark_sharpe: Threshold to test against.
        n_observations: Number of return observations.
        skewness_val: Skewness of returns.
        kurtosis_val: Regular kurtosis of returns (normal = 3.0).

    Returns:
        PSRResult.
    """
    se = sharpe_standard_error(observed_sharpe, skewness_val, kurtosis_val)
    if se < 1e-15:
        raise NumericalInstabilityError("Standard error is effectively zero, PSR is undefined")
    z = (observed_sharpe - benchmark_sharpe) * np.sqrt(n_observations - 1) / se
    prob = float(norm.cdf(z))
    return PSRResult(
        probability=prob,
        observed_sharpe=observed_sharpe,
        benchmark_sharpe=benchmark_sharpe,
        standard_error=se,
        skewness=skewness_val,
        kurtosis=kurtosis_val,
        n_observations=n_observations,
    )
