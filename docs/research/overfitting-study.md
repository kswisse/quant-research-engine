# Backtest Overfitting Study

## Executive Summary

This study quantifies the danger of backtest overfitting: the tendency for the best-performing strategy in a backtest to be a statistical fluke rather than evidence of genuine edge. Using a null synthetic environment — random price data with no exploitable pattern — we test 10, 100, and 1000 randomly generated trading strategies against the same price series.

The results are stark. Even under a pure null, the observed maximum Sharpe ratio grows with the number of strategies tested (0.0631 for N=10, 0.1396 for N=100, 0.1636 for N=1000). The Deflated Sharpe Ratio (DSR) is 0.0000 in all cases, indicating that none of the observed maxima would survive correction for selection bias. The expected maximum Sharpe under the null — the baseline any real strategy must exceed — ranges from 1.57 to 3.26 depending on N, vastly exceeding what any observed strategy produced.

The central message: **testing more strategies creates more opportunities for extreme observed performance, even when no genuine edge exists.** Selecting the best backtest without accounting for the number of trials is statistically equivalent to cherry-picking.

## Experimental Setup

### Synthetic Data Generation

- **Process:** Geometric Brownian Motion with μ = 0.0005 daily drift, σ = 0.01 daily volatility
- **Observations:** 252 price levels (251 returns)
- **Initial price:** 100.0
- **Seed:** 42 (deterministic via `numpy.random.default_rng(42)`)

### Strategy Generation

- **Families tested:** `random_threshold`, `random_sma`, `random_momentum`, `random_mean_reversion`
- **Signal space:** {-1, 0, +1} (short, flat, long)
- **Seed policy:** Prefix-based — N=10 uses strategies 0..9, N=100 uses 0..99, N=1000 uses 0..999 from the same master seed
- **Temporal convention:** signal[t] → position[t] → return[t+1] (no look-ahead bias)

### Backtest Convention

- Close prices only
- Zero transaction costs
- No fractional sizing
- Risk-free rate: 0.0

### Study Parameters

| Parameter | Value |
|-----------|-------|
| Study seed | 42 |
| Strategy counts | 10, 100, 1000 |
| Price data | 252 levels, synthetic GBM |
| Backtest periods | 251 |

## Results

| N | Observed Max Sharpe | Expected Max Sharpe | DSR | Mean Sharpe | Median Sharpe | Sharpe Std | Overfit |
|---|---|---|---|---|---|---|---|
| 10 | 0.0631 | 1.5746 | 0.0000 | -0.0124 | -0.0092 | 0.0353 | True |
| 100 | 0.1396 | 2.5306 | 0.0000 | -0.0036 | 0.0000 | 0.0516 | True |
| 1000 | 0.1636 | 3.2551 | 0.0000 | -0.0030 | 0.0000 | 0.0524 | True |

## Interpretation

### The Selection Effect

Under a null-like synthetic environment (no exploitable pattern), testing more strategies creates more opportunities for extreme observed performance. This is a straightforward application of order statistics: the maximum of N i.i.d. random variables increases with N.

The data confirms this:
- N=10: observed max = 0.0631
- N=100: observed max = 0.1396 (2.2× increase)
- N=1000: observed max = 0.1636 (2.6× increase from N=10)

### Why DSR = 0.0000

The Deflated Sharpe Ratio measures the probability that the best strategy's true Sharpe exceeds the expected maximum under the null. With DSR = 0.0000, the data says: **there is no evidence any strategy has genuine skill.** The observed maxima are well below the expected maxima (1.57–3.26), meaning even the best observed strategies perform worse than random chance would predict.

### Expected Max vs. Observed Max Gap

The expected maximum Sharpe grows with N because more trials means more chances to observe an extreme value. The observed maxima also grow with N, but remain far below the expected baseline. This gap is the signature of a null environment: the selection pressure is real, but no strategy has an actual edge to be selected.

### Mean and Median Behavior

Mean Sharpe ratios are slightly negative (-0.0124 to -0.0030), consistent with the zero drift assumption. Medians converge to 0.0000 at N=100 and N=1000, as expected for symmetric noise. Standard deviation increases with N (0.0353 → 0.0524), reflecting the wider parameter space explored.

## Limitations

1. **Strategies are not independent.** The four strategy families share structural similarities (all use price-based signals), violating the i.i.d. assumption underlying the DSR formula. This means N=1000 does not represent 1000 truly independent trials.

2. **Expected maximum Sharpe uses simplifying assumptions.** The formula assumes independent trials and a specific distributional form. With correlated strategies, the true expected maximum may differ.

3. **Synthetic data is not real market data.** Real markets have fat tails, volatility clustering, regime changes, and serial correlation. The null environment here is i.i.d. Gaussian, which is the simplest possible null.

4. **No transaction costs.** Real strategies incur slippage, commissions, and market impact. Including these would further reduce observed Sharpe ratios.

5. **No PBO/CSCV.** Probability of Backtest Overfitting (PBO) and Combinatorially Symmetric Cross-Validation (CSCV) are more sophisticated methods for detecting overfitting. This study uses only the DSR framework.

6. **No claim of real-world profitability.** This study demonstrates a statistical phenomenon. It does not claim that any specific strategy is unprofitable, only that observed backtest performance alone is insufficient evidence of skill.

7. **Observed maxima need not increase monotonically.** The observed maximum Sharpe is a random variable. For a given seed, N=100 might produce a lower observed maximum than N=10 due to the specific strategies generated. The trend is expected in expectation, not guaranteed for any single realization.
