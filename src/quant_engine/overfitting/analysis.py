"""Backtest overfitting analysis.

Analyzes an completed experiment to quantify the statistical danger
of selecting the best-performing strategy from many tested strategies.

This module consumes ExperimentResult — it does NOT regenerate strategies
or rerun backtests. It applies the existing statistical primitives
(expected_max_sharpe, DSR) to assess selection bias.
"""

from __future__ import annotations

import numpy as np

from quant_engine.experiments.results import ExperimentResult  # noqa: TC001
from quant_engine.overfitting.results import OverfittingAnalysisResult
from quant_engine.statistics.dsr import dsr_from_stats, expected_max_sharpe


def analyze_backtest_overfitting(
    experiment: ExperimentResult,
    variance_sr: float = 1.0,
) -> OverfittingAnalysisResult:
    """Analyze a completed experiment for backtest overfitting.

    Computes:
        A. Observed maximum Sharpe across all strategies
        B. Expected maximum Sharpe under the null of zero skill
        C. Deflated Sharpe Ratio (DSR) for the best strategy

    This does NOT:
        - Regenerate strategies
        - Rerun backtests
        - Select strategies
        - Claim statistical proof of anything

    Args:
        experiment: Completed experiment result.
        variance_sr: Variance of SR estimates across trials. Default 1.0.

    Returns:
        OverfittingAnalysisResult with descriptive and probabilistic analysis.

    Raises:
        ValueError: If experiment has zero strategies or invalid data.
    """
    results = experiment.strategy_results
    n = len(results)

    if n == 0:
        raise ValueError("Experiment has zero strategy results — cannot analyze")

    # Extract Sharpe ratios
    sharpes = np.array([r.sharpe_ratio for r in results])

    # Reject NaN/inf
    if not np.all(np.isfinite(sharpes)):
        raise ValueError("Experiment contains non-finite Sharpe ratios (NaN/inf)")

    # A. Observed maximum Sharpe
    max_idx = int(np.argmax(sharpes))
    observed_max = float(sharpes[max_idx])
    best_spec = results[max_idx].strategy_spec
    best_id = best_spec.strategy_id

    # B. Expected maximum Sharpe under the null
    expected_max = expected_max_sharpe(n, variance_sr)

    # C. DSR for the best strategy
    best = results[max_idx]
    track_length = best.n_observations

    if track_length < 2:
        raise ValueError(
            f"Best strategy has only {track_length} observations — "
            "need at least 2 for DSR computation"
        )

    dsr_result = dsr_from_stats(
        observed_sharpe=observed_max,
        n_observations=track_length,
        skewness_val=best.skewness,
        kurtosis_val=best.kurtosis,
        n_trials=n,
        variance_sr=variance_sr,
    )

    # Descriptive statistics
    mean_s = float(np.mean(sharpes))
    median_s = float(np.median(sharpes))
    std_s = float(np.std(sharpes, ddof=1)) if n > 1 else 0.0

    return OverfittingAnalysisResult(
        experiment_id=experiment.experiment_id,
        num_trials=n,
        observed_max_sharpe=observed_max,
        best_strategy_id=best_id,
        expected_max_sharpe=expected_max,
        deflated_sharpe=dsr_result.probability,
        is_overfit=dsr_result.is_overfit,
        mean_sharpe=mean_s,
        median_sharpe=median_s,
        sharpe_std=std_s,
        performance_summary=experiment.performance_summary.model_dump(),
        dsr_result=dsr_result,
        variance_sr=variance_sr,
        track_record_length=track_length,
    )
