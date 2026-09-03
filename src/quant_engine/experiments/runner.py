"""Experiment runner.

Orchestrates: StrategyGenerator → BacktestEngine → ExperimentResult.
Preserves the architectural separation between generation and evaluation.
"""

from __future__ import annotations

import numpy as np

from quant_engine.backtest.data import PriceData
from quant_engine.backtest.engine import run_backtest
from quant_engine.backtest.generator import StrategyGenerator
from quant_engine.experiments.config import ExperimentConfig  # noqa: TC001
from quant_engine.experiments.results import ExperimentResult, PerformanceSummary, StrategyResult
from quant_engine.statistics.descriptive import kurtosis as compute_kurt
from quant_engine.statistics.descriptive import skewness as compute_skew


def _compute_performance_summary(results: list[StrategyResult]) -> PerformanceSummary:
    """Compute aggregate performance distribution from per-strategy results."""
    sharpes = np.array([r.sharpe_ratio for r in results])
    returns = np.array([r.total_return for r in results])

    return PerformanceSummary(
        n_strategies=len(results),
        sharpe_min=float(np.min(sharpes)),
        sharpe_max=float(np.max(sharpes)),
        sharpe_mean=float(np.mean(sharpes)),
        sharpe_median=float(np.median(sharpes)),
        sharpe_std=float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0,
        sharpe_q5=float(np.percentile(sharpes, 5)),
        sharpe_q25=float(np.percentile(sharpes, 25)),
        sharpe_q75=float(np.percentile(sharpes, 75)),
        sharpe_q95=float(np.percentile(sharpes, 95)),
        return_min=float(np.min(returns)),
        return_max=float(np.max(returns)),
        return_mean=float(np.mean(returns)),
    )


class ExperimentRunner:
    """Executes a reproducible experiment.

    Flow:
        ExperimentConfig
            → StrategyGenerator.generate()
            → StrategyGenerator.create_strategy()
            → run_backtest()
            → StrategyResult[]
            → PerformanceSummary
            → ExperimentResult
    """

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a complete experiment.

        Args:
            config: Experiment configuration.

        Returns:
            Complete experiment result with per-strategy results and summary.

        Raises:
            ValueError: If config is invalid or data is insufficient.
        """
        # Build PriceData
        data = PriceData(np.array(config.price_data, dtype=np.float64))

        # Generate strategies
        generator = StrategyGenerator(
            seed=config.seed,
            generator_version=config.generator_version,
        )
        specs = generator.generate(config.num_strategies)

        # Evaluate each strategy
        strategy_results: list[StrategyResult] = []
        for spec in specs:
            strategy = generator.create_strategy(spec)
            output = run_backtest(data, strategy)

            rets = output.strategy_returns
            n_obs = len(rets)

            # Compute descriptive metrics
            mean_ret = float(np.mean(rets))
            vol = float(np.std(rets, ddof=1)) if n_obs > 1 else 0.0

            # Sharpe ratio (periodic, non-annualized)
            sharpe = mean_ret / vol if vol > 1e-15 else 0.0

            # Skewness and kurtosis
            sk = 0.0
            ku = 3.0
            if n_obs >= 3:
                try:
                    sk = compute_skew(rets)
                except Exception:
                    sk = 0.0
            if n_obs >= 4:
                try:
                    ku = compute_kurt(rets)
                except Exception:
                    ku = 3.0

            strategy_results.append(
                StrategyResult(
                    strategy_spec=spec,
                    sharpe_ratio=sharpe,
                    total_return=output.total_return,
                    n_observations=n_obs,
                    skewness=sk,
                    kurtosis=ku,
                    mean_return=mean_ret,
                    volatility=vol,
                )
            )

        summary = _compute_performance_summary(strategy_results)

        return ExperimentResult(
            config=config,
            strategy_results=strategy_results,
            performance_summary=summary,
        )
