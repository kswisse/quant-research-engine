# Statistical Engine — Research Documentation

This document defines the mathematical foundations implemented in `src/quant_engine/statistics/`.

**Reference:** Bailey, D.H. & López de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality." *Journal of Portfolio Management*, 40(5), 94-107.

---

## 1. Sharpe Ratio

### Purpose

Measures risk-adjusted return: how much excess return per unit of volatility.

### Formula

```
SR = mean(r - rf) / std(r - rf)
```

where:
- `r` = periodic decimal returns (0.01 = 1%)
- `rf` = periodic risk-free rate (default 0.0)
- `std` uses T-1 denominator (sample standard deviation, Bessel's correction)

### Annualization

```
SR_annual = SR_periodic * sqrt(periods_per_year)
```

Annualization requires an explicit `periods_per_year` parameter. It is never applied implicitly.

### Assumptions

- Returns are independent and identically distributed (i.i.d.) — not required for the computation, but required for statistical inference (PSR/DSR).
- Risk-free rate is constant over the period.

### Edge Cases

| Input | Behavior |
|-------|----------|
| Empty series | `InsufficientDataError` |
| Single observation | `InsufficientDataError` (need ≥ 2 for std) |
| Zero volatility | `NumericalInstabilityError` (undefined) |
| All-zero returns | SR = 0 |

### Units

Dimensionless (decimal return per decimal volatility).

---

## 2. Descriptive Statistics

### Mean

```
r_bar = (1/T) * sum(r_t)
```

### Standard Deviation

```
s = sqrt( (1/(T-1)) * sum((r_t - r_bar)^2) )
```

Uses T-1 denominator (sample standard deviation, Bessel's correction). This is the convention used in Sharpe ratio calculations.

### Skewness (Fisher's)

```
g1 = (T / ((T-1)*(T-2))) * (1/T) * sum(((r_t - r_bar) / s)^3)
```

This is the adjusted Fisher-Pearson skewness coefficient. A normal distribution has skewness = 0.

### Kurtosis (Regular, Not Excess)

```
K = (1/T) * sum(((r_t - r_bar) / s)^4)
```

Bias-corrected form:

```
K_adj = K + (T-1) / ((T-2)*(T-3)) * ((T+1)*K - 3*(T-1))
```

**Convention:** This returns REGULAR kurtosis (normal = 3.0), NOT excess kurtosis.

The Bailey & López de Prado PSR/DSR formulas use γ₄ as regular kurtosis. The excess kurtosis adjustment appears inside the formula as (γ₄ - 1)/4.

---

## 3. Standard Error of the Sharpe Ratio

### Formula (Lo 2002, Bailey & López de Prado 2012)

```
se = sqrt( 1 - γ₃ * SR + (γ₄ - 1)/4 * SR² )
```

where:
- SR = observed Sharpe ratio
- γ₃ = skewness of returns
- γ₄ = regular kurtosis (normal = 3.0)

### Interpretation

For normal returns (γ₃ = 0, γ₄ = 3):
```
se = sqrt(1 + SR²/2)
```

Negative skewness or high kurtosis increases the standard error, making the Sharpe ratio less reliable.

### Numerical Stability

The argument under the square root can become negative for extreme skewness/SR combinations. This raises `NumericalInstabilityError`.

---

## 4. Probabilistic Sharpe Ratio (PSR)

### Purpose

Computes P(true SR > SR*), the probability that the true Sharpe ratio exceeds a benchmark.

### Formula

```
PSR(SR*) = Φ( (SR̂ - SR*) * sqrt(T-1) / se )
```

where:
- Φ = standard normal CDF
- SR̂ = observed Sharpe ratio
- SR* = benchmark Sharpe ratio
- T = number of observations
- se = standard error (see above)

### Output

A probability in [0, 1].

### Assumptions

- Returns are i.i.d.
- The Sharpe ratio estimator is approximately normally distributed (corrected by Lo 2002 for non-normality).
- T is sufficiently large for the asymptotic approximation.

### Validation

| Case | Expected Behavior |
|------|-------------------|
| SR̂ = SR* | PSR ≈ 0.5 |
| SR̂ >> SR* | PSR → 1 |
| SR̂ << SR* | PSR → 0 |
| Larger T | Higher confidence (if SR̂ > SR*) |

---

## 5. Expected Maximum Sharpe Ratio

### Purpose

Estimates the expected maximum Sharpe ratio from N independent trials under the null hypothesis of zero skill.

### Formula (Bailey & López de Prado 2014, Eq. 6)

```
SR₀ = √V * [ (1 - γ) * Φ⁻¹(1 - 1/N)
            + γ * Φ⁻¹(1 - 1/(N*e)) ]
```

where:
- V = variance of Sharpe estimates across trials
- γ = Euler-Mascheroni constant ≈ 0.5772156649
- Φ⁻¹ = inverse standard normal CDF (quantile function)
- N = number of independent trials
- e = Euler's number

### Assumptions

- Trials are independent.
- Sharpe estimates are approximately normally distributed.
- The approximation is asymptotic (improves with large N).

### Interpretation

This is the "winner's curse" — the Sharpe ratio you would expect from the best of N random strategies even if none have genuine skill.

### Known Limitations

- The formula is an asymptotic approximation; accuracy decreases for very small N (N < 5).
- Assumes independence between trials. Correlated trials require adjustment (see Appendix 3 of Bailey & López de Prado 2014).

---

## 6. Deflated Sharpe Ratio (DSR)

### Purpose

Evaluates whether an observed best Sharpe remains statistically significant after accounting for selection among multiple trials.

### Formula

DSR = PSR(SR₀) where SR₀ is the deflated benchmark:

```
DSR = Φ( (SR̂ - SR₀) * sqrt(T-1) / se )
```

where:
- SR̂ = observed Sharpe ratio
- SR₀ = expected maximum Sharpe from N trials (see above)
- T = number of observations
- se = standard error (see above)

### Inputs

| Input | Description |
|-------|-------------|
| `returns` | 1-D array of decimal returns |
| `n_trials` | Number of independent strategies tested |
| `variance_sr` | Variance of SR estimates across trials (default 1.0) |

### Output

`DSRResult` containing:
- `probability`: The DSR value in [0, 1]
- `observed_sharpe`: SR̂
- `benchmark_sharpe`: SR₀
- `standard_error`: se
- `skewness`, `kurtosis`: Return distribution moments
- `n_observations`, `n_trials`, `variance_sr`: Inputs
- `is_overfit`: True if probability < 0.5

### Interpretation

- DSR > 0.95: Strong evidence the strategy is not a false discovery
- DSR > 0.5: Strategy survives deflation (but modest confidence)
- DSR < 0.5: Strategy likely a product of selection bias

### Key Insight

DSR corrects for TWO sources of inflation:
1. **Non-normality** — via the standard error adjustment (Lo 2002)
2. **Selection bias** — via the deflated benchmark (multiple testing)

### Known Limitations

- Does not detect look-ahead bias, survivorship bias, or cost-modeling errors.
- Assumes trial independence; correlated trials require adjusted N.
- The expected maximum is an approximation; exact computation would require integration.
- Does not account for data-mining within a single strategy's parameter space (PBO addresses this separately).

---

## 7. Conventions Summary

| Convention | Choice | Rationale |
|------------|--------|-----------|
| Kurtosis | Regular (normal = 3.0) | Matches Bailey & López de Prado notation |
| Std denominator | T-1 (Bessel's correction) | Standard for sample statistics |
| Annualization | Explicit periods_per_year | Prevents silent assumptions |
| Risk-free rate | Default 0.0 | Excess returns = raw returns by default |
| Numerical stability | Raise error on instability | Fail loudly rather than return misleading values |
| Mathematical notation | T, N for sample size, trial count | Standard in quantitative finance literature |
