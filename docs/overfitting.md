# Backtest Overfitting Analysis

Quantifies the statistical danger of selecting the best-performing strategy from many tested strategies.

## The Problem

When you test N strategies and pick the best one, you benefit from selection bias — even if all strategies are random. The more you test, the better your "best" looks, creating an illusion of skill.

**Key question:** Is the best strategy's Sharpe ratio genuinely better than what pure chance would produce?

## Architecture

```
ExperimentResult
       ↓
analyze_backtest_overfitting()
       ↓
OverfittingAnalysisResult
```

## Components

### OverfittingAnalysisResult

The complete analysis output:

- `experiment_id` — Link to the source experiment
- `num_trials` — Number of strategies tested (N)
- `observed_max_sharpe` — Best Sharpe ratio observed
- `best_strategy_id` — ID of the winning strategy (observational only)
- `expected_max_sharpe` — Expected best under the null of zero skill
- `deflated_sharpe` — DSR probability that true SR > expected max
- `is_overfit` — True if DSR < 0.5
- `mean_sharpe`, `median_sharpe`, `sharpe_std` — Descriptive statistics
- `track_record_length` — Number of observations for the best strategy
- `variance_sr` — Variance of SR estimates across trials

### analyze_backtest_overfitting()

```python
from quant_engine.overfitting import analyze_backtest_overfitting

experiment = runner.run(config)
analysis = analyze_backtest_overfitting(experiment)

print(f"Best Sharpe: {analysis.observed_max_sharpe:.4f}")
print(f"Expected Max: {analysis.expected_max_sharpe:.4f}")
print(f"DSR: {analysis.deflated_sharpe:.4f}")
print(f"Overfit: {analysis.is_overfit}")
```

## Statistical Framework

### Expected Maximum Sharpe (Lo, 2002)

Under the null hypothesis that all strategies have zero true Sharpe:

```
E[max(SR₁, ..., SRₙ)] ≈ √(2 ln N) × √variance_sr
```

With `variance_sr = 1.0` (default), this equals `√(2 ln N)`.

**This is the baseline.** If your best strategy's Sharpe equals this, it could be pure selection bias.

### Deflated Sharpe Ratio (Bailey & López de Prado, 2014)

The DSR answers: "What is the probability that the best strategy's true Sharpe exceeds the expected maximum under the null?"

```
DSR = P(SR* > E[max])
```

Where SR* is the estimated Sharpe of the best strategy, and E[max] is the expected maximum under zero skill.

The DSR incorporates:
- The number of strategies tested (N)
- The skewness and kurtosis of returns
- The track record length (T)
- The variance of SR estimates across trials

### Interpretation

| DSR Range | Interpretation |
|-----------|---------------|
| DSR > 0.95 | Strong evidence of genuine performance |
| 0.5 < DSR < 0.95 | Weak evidence; more data needed |
| DSR < 0.5 | Evidence of selection bias (overfitting) |

## Independence Assumption

The analysis uses the nominal trial count N. This is a conservative assumption because:

1. **Correlated strategies** — Strategies from the same family produce correlated signals
2. **Effective N < nominal N** — The true number of independent tests is smaller

This means the DSR is *conservative* — it may flag strategies as overfit when they're actually OK, but it won't miss genuinely overfit strategies. This is the safe direction for risk management.

## Limitations

1. **Descriptive, not causal** — The analysis quantifies statistical danger; it doesn't prove overfitting
2. **No strategy selection** — It doesn't choose strategies; it reports on the experiment
3. **No PBO/CSCV** — The Probability of Backtest Overfitting is a separate framework deferred to future phases
4. **Single path** — The current analysis doesn't account for strategy paths or time-dependent selection

## Integration

```python
from quant_engine.experiments import ExperimentRunner, ExperimentConfig
from quant_engine.overfitting import analyze_backtest_overfitting

# Run experiment
config = ExperimentConfig(seed=42, num_strategies=100, price_data=prices)
experiment = ExperimentRunner().run(config)

# Analyze overfitting
analysis = analyze_backtest_overfitting(experiment)

# Interpret
if analysis.is_overfit:
    print("WARNING: Best strategy likely overfit")
    print(f"DSR = {analysis.deflated_sharpe:.4f} < 0.5")
else:
    print(f"DSR = {analysis.deflated_sharpe:.4f} — weak to moderate evidence")
```
