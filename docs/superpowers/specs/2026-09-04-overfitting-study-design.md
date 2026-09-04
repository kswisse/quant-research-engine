# Phase 0.7 — Research Demonstration & Overfitting Study

## Objective

Turn existing components into a reproducible quantitative research demonstration showing how multiple testing creates apparently strong strategies from noise.

## Architecture

```
OverfittingStudyConfig
        ↓
OverfittingStudy.run()
        ↓
  for each N in strategy_counts:
        ↓
    ExperimentRunner.run(ExperimentConfig)
        ↓
    ExperimentResult
        ↓
    analyze_backtest_overfitting()
        ↓
    OverfittingAnalysisResult
        ↓
StudyResult
```

## Components

### OverfittingStudyConfig

- `seed: int` — master seed
- `strategy_counts: tuple[int, ...]` — e.g. (10, 100, 1000)
- `price_data: list[float]` — synthetic prices
- Deterministic `study_id` via SHA-256 of canonical JSON

### OverfittingStudyResult

- `study_id: str`
- `config: OverfittingStudyConfig`
- `analyses: list[OverfittingAnalysisResult]` — one per N

### run_overfitting_study()

Orchestrates: for each N, create ExperimentConfig(seed, N, prices), run experiment, analyze overfitting.

## Seed Policy

Prefix-based: N=10 uses strategies 0..9, N=100 uses 0..99, N=1000 uses 0..999. Same seed + same prices = same results.

## Synthetic Data

IID normal returns: r_t ~ N(mu, sigma). No exploitable structure. Controlled null environment.

## Files

- `src/quant_engine/research/__init__.py` — public API
- `src/quant_engine/research/overfitting_study.py` — config, result, runner
- `tests/test_research.py` — comprehensive tests
- `docs/research/overfitting-study.md` — methodology and results

## Testing

- Config validation, serialization, deterministic ID
- Study execution for N=10, 100, 1000
- Same price data across experiments
- Reproducibility (run twice, compare)
- Different seeds produce different results
- Expected max Sharpe increases with N
- Serialization roundtrip
- Invalid inputs rejected

## Scope Boundary

No real market data, no PBO/CSCV, no optimization, no transaction costs. Research demonstration only.
