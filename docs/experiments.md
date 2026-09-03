# Experiment Runner

Bridges strategy generation and evaluation in a reproducible pipeline.

## Architecture

```
ExperimentConfig
       ↓
StrategyGenerator
       ↓
StrategySpec[]
       ↓
StrategyGenerator.create_strategy()
       ↓
BacktestEngine
       ↓
StrategyResult[]
       ↓
PerformanceSummary
       ↓
ExperimentResult
```

## Key Components

### ExperimentConfig

Immutable, validated configuration:

```python
config = ExperimentConfig(
    seed=42,                    # master seed for reproducibility
    num_strategies=100,         # how many strategies to generate
    price_data=[100.0, ...],    # closing prices for backtesting
    generator_version="1",      # version tracking
    name="my_experiment",       # optional label
)
```

Deterministic `experiment_id` derived from canonical JSON of all fields.

### StrategyResult

Per-strategy result linking spec to performance:

```python
result.strategy_spec    # StrategySpec (family, params, seed, id)
result.sharpe_ratio     # periodic Sharpe (non-annualized)
result.total_return     # cumulative return
result.n_observations   # number of return periods
result.skewness         # return distribution skewness
result.kurtosis         # return distribution kurtosis
result.mean_return      # arithmetic mean of returns
result.volatility       # sample standard deviation of returns
```

### PerformanceSummary

Aggregate distribution across all strategies:

```python
summary.sharpe_min / sharpe_max / sharpe_mean / sharpe_median / sharpe_std
summary.sharpe_q5 / sharpe_q25 / sharpe_q75 / sharpe_q95
summary.return_min / return_max / return_mean
```

### ExperimentRunner

```python
runner = ExperimentRunner()
result = runner.run(config)
```

## Reproducibility

Same `ExperimentConfig` + same `price_data` = identical results:

```python
r1 = runner.run(config)
r2 = runner.run(config)
assert r1.experiment_id == r2.experiment_id
assert r1.sharpe_ratios == r2.sharpe_ratios
```

## StrategySpec ↔ BacktestResult Mapping

Each `StrategyResult` contains the full `StrategySpec`, so a researcher can:

- Identify which strategy produced which result
- Reconstruct any strategy from its spec
- Trace from experiment → strategy → performance

## Why Generation and Evaluation Remain Separate

The generator never sees:
- PriceData
- BacktestResult
- Sharpe ratios
- Performance rankings

This prevents hidden optimization. The generator produces strategies from fixed distributions; the runner evaluates them independently.

## Performance Metrics

Phase 0.5 collects only descriptive metrics:
- Sharpe ratio (periodic, non-annualized)
- Total return
- Mean return, volatility, skewness, kurtosis

**Not implemented yet:**
- DSR (Deflated Sharpe Ratio)
- PSR (Probabilistic Sharpe Ratio)
- PBO (Probability of Backtest Overfitting)
- Multiple-testing correction
- Strategy selection

This is descriptive backtesting, not statistical significance analysis.

## Limitations

- No transaction costs
- Close prices only
- No annualization (explicit periods_per_year needed)
- Simple descriptive statistics only
- No walk-forward or out-of-sample testing
