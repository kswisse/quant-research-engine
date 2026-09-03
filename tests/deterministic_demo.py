"""Deterministic demonstration of Sharpe, PSR, and DSR.

This script verifies that the statistical foundation works correctly
using a synthetic return series. It is NOT a trading strategy and
does NOT claim any real-world profitability.

Run: python -m tests.deterministic_demo
Or:  python tests/deterministic_demo.py
"""

import numpy as np

from quant_engine.statistics.descriptive import (
    excess_kurtosis,
    kurtosis,
    mean,
    skewness,
    std,
)
from quant_engine.statistics.dsr import dsr, expected_max_sharpe
from quant_engine.statistics.psr import psr
from quant_engine.statistics.sharpe import sharpe_ratio


def main() -> None:
    print("=" * 60)
    print("Quantitative Research Engine — Statistical Foundation Demo")
    print("=" * 60)
    print()
    print("This is a SYNTHETIC demonstration. No real market data.")
    print("No trading strategy is being evaluated.")
    print()

    # Generate deterministic synthetic returns
    rng = np.random.default_rng(seed=42)
    returns = rng.normal(loc=0.0005, scale=0.01, size=252)

    print(f"Synthetic return series: {len(returns)} observations")
    print(f"  Generated with seed=42, N(0.0005, 0.01)")
    print()

    # Descriptive statistics
    print("--- Descriptive Statistics ---")
    print(f"  Mean:              {mean(returns):.6f}")
    print(f"  Std (T-1):         {std(returns):.6f}")
    print(f"  Skewness:          {skewness(returns):.6f}")
    print(f"  Kurtosis:          {kurtosis(returns):.6f}")
    print(f"  Excess Kurtosis:   {excess_kurtosis(returns):.6f}")
    print()

    # Sharpe Ratio
    sr = sharpe_ratio(returns)
    sr_annual = sharpe_ratio(returns, periods_per_year=252)
    print("--- Sharpe Ratio ---")
    print(f"  Periodic SR:       {sr:.6f}")
    print(f"  Annualized SR:     {sr_annual:.6f}")
    print()

    # PSR
    psr_result = psr(returns, benchmark_sharpe=0.0)
    print("--- Probabilistic Sharpe Ratio (PSR) ---")
    print(f"  Benchmark SR:      {psr_result.benchmark_sharpe:.6f}")
    print(f"  Observed SR:       {psr_result.observed_sharpe:.6f}")
    print(f"  Standard Error:    {psr_result.standard_error:.6f}")
    print(f"  P(true SR > 0):    {psr_result.probability:.6f}")
    print()

    # Expected Maximum Sharpe
    print("--- Expected Maximum Sharpe (Multiple Testing) ---")
    for n in [10, 100, 1000]:
        em = expected_max_sharpe(n, variance_sr=1.0)
        print(f"  N={n:>5d} trials:  E[max SR] = {em:.4f}")
    print()

    # DSR
    dsr_result = dsr(returns, n_trials=100, variance_sr=1.0)
    print("--- Deflated Sharpe Ratio (DSR) ---")
    print(f"  Number of trials:  {dsr_result.n_trials}")
    print(f"  Variance of SR:    {dsr_result.variance_sr}")
    print(f"  Benchmark (SR_0):  {dsr_result.benchmark_sharpe:.6f}")
    print(f"  Observed SR:       {dsr_result.observed_sharpe:.6f}")
    print(f"  DSR probability:   {dsr_result.probability:.6f}")
    print(f"  Is overfit:        {dsr_result.is_overfit}")
    print()

    print("=" * 60)
    print("Verification: All computations completed without error.")
    print("This demonstrates the statistical engine is functional.")
    print("=" * 60)


if __name__ == "__main__":
    main()
