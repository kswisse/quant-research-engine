"""Deflated Sharpe Ratio (DSR).

Reference:
    Bailey, D.H. & López de Prado, M. (2014).
    "The Deflated Sharpe Ratio: Correcting for Selection Bias,
     Backtest Overfitting and Non-Normality".
    Journal of Portfolio Management, 40(5), 94-107.

The DSR is a PSR where the benchmark is the expected maximum Sharpe ratio
arising from multiple independent trials under the null hypothesis of zero skill.

Mathematical formula:

    1. Expected Maximum Sharpe (under null of zero skill):

        SR₀ = √V * [ (1 - γ) * Φ⁻¹(1 - 1/N)
                    + γ * Φ⁻¹(1 - 1/(N*e)) ]

    where:
        V = variance of the Sharpe estimates across trials
        γ = Euler-Mascheroni constant ≈ 0.5772156649
        Φ⁻¹ = inverse standard normal CDF (quantile function)
        N = number of independent trials
        e = Euler's number

    2. DSR = PSR(SR₀) where SR₀ is the deflated benchmark:

        DSR = Φ( (SR̂ - SR₀) * √(T-1) / se )

    where se is the standard error of the Sharpe estimator:

        se = √( 1 - γ₃ * SR̂ + (γ₄ - 1)/4 * SR̂² )

Conventions (explicitly chosen from Bailey & López de Prado 2014):
    - kurtosis (γ₄) is REGULAR kurtosis, not excess. Normal = 3.0.
    - The formula internally computes (γ₄ - 1)/4 as the excess kurtosis term.
    - When γ₄ = 3 (normal), se = √(1 + SR̂²/2).
    - V (variance of SR estimates) is an input, not estimated from data.
      If V is unknown, a common assumption is V = 1 (standardized trials).
    - The expected maximum is an asymptotic approximation (improves with large N).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from quant_engine.statistics.descriptive import kurtosis as compute_kurtosis
from quant_engine.statistics.descriptive import skewness, validate_returns
from quant_engine.statistics.psr import psr_from_stats
from quant_engine.statistics.sharpe import sharpe_ratio

EULER_MASCHERONI = 0.5772156649015329


@dataclass(frozen=True)
class DSRResult:
    """Result of a Deflated Sharpe Ratio computation.

    Attributes:
        probability: P(true SR > SR₀), in [0, 1]. The DSR itself.
        observed_sharpe: The estimated Sharpe ratio from the sample.
        benchmark_sharpe: The deflated threshold (SR₀) from multiple testing.
        expected_max_sharpe: Alias for benchmark_sharpe (SR₀).
        standard_error: Standard error of the Sharpe estimator.
        skewness: Skewness of the return series.
        kurtosis: Regular kurtosis (normal = 3.0).
        n_observations: Number of return observations.
        n_trials: Number of independent strategies tested.
        variance_sr: Variance of SR estimates across trials.
        is_overfit: True if DSR < 0.5 (evidence of selection bias).
    """

    probability: float
    observed_sharpe: float
    benchmark_sharpe: float
    expected_max_sharpe: float
    standard_error: float
    skewness: float
    kurtosis: float
    n_observations: int
    n_trials: int
    variance_sr: float
    is_overfit: bool


def expected_max_sharpe(
    n_trials: int,
    variance_sr: float = 1.0,
) -> float:
    """Expected maximum Sharpe ratio under multiple testing (asymptotic approximation).

    Formula (Bailey & López de Prado 2014, Eq. 6, Appendix 1):

        SR₀ = √V * [ (1 - γ) * Φ⁻¹(1 - 1/N)
                    + γ * Φ⁻¹(1 - 1/(N*e)) ]

    where:
        V = variance of Sharpe estimates across trials
        γ = Euler-Mascheroni constant ≈ 0.5772156649
        Φ⁻¹ = inverse standard normal CDF
        N = number of independent trials
        e = Euler's number

    This is the expected maximum of N independent Sharpe estimates drawn
    from N(0, V). It represents the "winner's curse" — the Sharpe ratio
    you would expect from the best of N random strategies even if none
    have genuine skill.

    Assumptions:
        - Trials are independent.
        - Sharpe estimates are approximately normally distributed.
        - The approximation improves with larger N.

    Args:
        n_trials: Number of independent strategies tested (N ≥ 1).
        variance_sr: Variance of the SR estimates across trials. Default 1.0.

    Returns:
        Expected maximum Sharpe ratio (always ≥ 0 for valid inputs).

    Raises:
        ValueError: If n_trials < 1 or variance_sr < 0.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if variance_sr < 0:
        raise ValueError(f"variance_sr must be >= 0, got {variance_sr}")
    if n_trials == 1:
        return 0.0
    emc = EULER_MASCHERONI
    z = (1 - emc) * norm.ppf(1 - 1.0 / n_trials) + emc * norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(variance_sr) * z)


def dsr(
    returns: np.ndarray,
    n_trials: int,
    variance_sr: float = 1.0,
) -> DSRResult:
    """Compute the Deflated Sharpe Ratio.

    Args:
        returns: 1-D array of decimal returns.
        n_trials: Number of independent strategies tested.
        variance_sr: Variance of SR estimates across trials. Default 1.0.

    Returns:
        DSRResult with the probability that true SR > expected max SR from N trials.

    Raises:
        ValueError: If inputs are invalid.
        NumericalInstabilityError: If computation is numerically unstable.
    """
    r = validate_returns(returns)
    sr_hat = sharpe_ratio(r)
    s3 = skewness(r)
    s4 = compute_kurtosis(r)  # Regular kurtosis, normal = 3.0
    T = len(r)  # noqa: N806 — mathematical notation for sample size
    sr0 = expected_max_sharpe(n_trials, variance_sr)
    psr_result = psr_from_stats(sr_hat, sr0, T, s3, s4)
    return DSRResult(
        probability=psr_result.probability,
        observed_sharpe=sr_hat,
        benchmark_sharpe=sr0,
        expected_max_sharpe=sr0,
        standard_error=psr_result.standard_error,
        skewness=s3,
        kurtosis=s4,
        n_observations=T,
        n_trials=n_trials,
        variance_sr=variance_sr,
        is_overfit=psr_result.probability < 0.5,
    )


def dsr_from_stats(
    observed_sharpe: float,
    n_observations: int,
    skewness_val: float,
    kurtosis_val: float,
    n_trials: int,
    variance_sr: float = 1.0,
) -> DSRResult:
    """Compute DSR from pre-computed statistics (lower-level interface).

    Useful for testing or when statistics are already available.

    Args:
        observed_sharpe: Estimated Sharpe ratio.
        n_observations: Number of return observations.
        skewness_val: Skewness of returns.
        kurtosis_val: Regular kurtosis of returns (normal = 3.0).
        n_trials: Number of independent strategies tested.
        variance_sr: Variance of SR estimates across trials.

    Returns:
        DSRResult.
    """
    sr0 = expected_max_sharpe(n_trials, variance_sr)
    psr_result = psr_from_stats(observed_sharpe, sr0, n_observations, skewness_val, kurtosis_val)
    return DSRResult(
        probability=psr_result.probability,
        observed_sharpe=observed_sharpe,
        benchmark_sharpe=sr0,
        expected_max_sharpe=sr0,
        standard_error=psr_result.standard_error,
        skewness=skewness_val,
        kurtosis=kurtosis_val,
        n_observations=n_observations,
        n_trials=n_trials,
        variance_sr=variance_sr,
        is_overfit=psr_result.probability < 0.5,
    )
