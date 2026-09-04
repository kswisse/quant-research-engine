# Quant Research Engine — Architecture

## Purpose

The Quant Research Engine is a modular monolith for quantitative research. It supports four major research systems:

1. **Backtest Overfitting Detector** — detect when repeated strategy experimentation produces false discoveries
2. **Prediction Market Mispricing Engine** — detect pricing inefficiencies across prediction markets
3. **Breeden–Litzenberger Risk-Neutral Distribution** — extract risk-neutral densities from options data
4. **Hawkes Information Diffusion Model** — model the relationship between information events and market movements

## Current Phase

Phase 0: Foundation + Backtest Overfitting Detector (System D)

Systems A, B, and C will be implemented in future phases.

## Architecture: Modular Monolith

The project is a **modular monolith** — a single deployable package with clear internal boundaries, not a collection of microservices.

### Dependency Direction (Dependency Inversion)

```
    applications / research workflows
                 │
                 ▼
         domain abstractions  (types, interfaces, protocols)
                 │
                 ▼
         data interfaces       (protocols for storage, ingestion)
                 │
                 ▼
         concrete implementations  (parquet, sqlite, API clients)
```

**The critical rule:** Core/domain code must NOT depend on concrete database, API, filesystem, or frontend implementations. High-level quantitative logic depends only on stable abstractions.

### Layer Responsibilities

| Layer | Location | Responsibilities | Dependencies |
|-------|----------|-----------------|--------------|
| **Core** | `src/quant_engine/core/` | Configuration, error types, reproducibility, seed control | None (leaf) |
| **Statistics** | `src/quant_engine/statistics/` | Statistical methods: Sharpe, PSR, DSR, variance analysis | Core |
| **Backtest** | `src/quant_engine/backtest/` | Strategy generation, backtesting, overfitting detection | Core, Statistics |
| **Data** | `src/quant_engine/data/` (future) | Schema definitions, data ingestion, storage | Core |
| **Prediction** | `src/quant_engine/prediction/` (future) | Prediction market analysis | Core, Data |
| **Options** | `src/quant_engine/options/` (future) | Options analytics, risk-neutral density | Core, Statistics, Data |
| **Diffusion** | `src/quant_engine/diffusion/` (future) | Hawkes processes, event modeling | Core, Statistics, Data |
| **Research** | `research/` | Experiment tracking, reproducibility enforcement | All |
| **API** | (future) | REST/HTTP interface | All |

### Why This Structure

1. **Correctness first** — Quantitative code must be verifiable. Clear boundaries make testing possible.
2. **Reproducibility** — Seed control in Core ensures deterministic behavior.
3. **Independence** — Each system (A-D) can be developed and tested in isolation.
4. **Future extensibility** — Adding a new system means adding a new directory, not modifying existing ones.

## Technology Decisions

### Why Python 3.12+
- Modern type hints, performance improvements, the ecosystem for quantitative work.

### Why Pydantic
- Validation at boundaries, immutable data structures, configuration management.

### Why NumPy/SciPy (not pandas)
- NumPy is the foundation for all numerical work. SciPy provides statistical functions.
- Polars is used for data processing (not pandas) due to performance and correctness advantages.
- pandas is not needed for the core quantitative logic.

### Why PostgreSQL is Deferred
- Phase 0 uses in-memory DataFrames and Parquet files.
- PostgreSQL will be added when multi-user persistence or complex queries are needed.

### Why Docker is Deferred
- Development happens locally. Docker adds complexity without current benefit.
- Docker will be added when reproducible environments are needed for deployment.

### Why Microservices are Deferred
- A monolith with clear internal boundaries is simpler, faster to develop, and easier to debug.
- Microservices will be considered only when scaling demands it (likely post-SaaS).

## Testing Strategy

| Test Type | Location | Purpose |
|-----------|----------|---------|
| Unit | `tests/unit/` | Individual functions and classes |
| Property-based | `tests/property/` | Statistical invariants, numerical properties |
| Numerical | `tests/numerical/` (future) | Known analytical cases, tolerances |
| Integration | `tests/integration/` (future) | Data → model → output pipelines |

## Data Flow (Future)

```
Raw Data (API/Parquet)
    ↓
Validated Schemas (Pydantic)
    ↓
Domain Models (immutable)
    ↓
Statistical Computation
    ↓
Research Output (ExperimentResult)
    ↓
Persistence (Parquet/PostgreSQL)
```

## Statistical Engine

The statistics module (`src/quant_engine/statistics/`) provides the mathematical foundation for backtest overfitting detection.

### Implemented Metrics

| Metric | Module | Purpose |
|--------|--------|---------|
| **Sharpe Ratio** | `sharpe.py` | Risk-adjusted return measure |
| **PSR** | `psr.py` | Probability that true SR > benchmark |
| **DSR** | `dsr.py` | Sharpe corrected for selection bias |
| **Expected Max SR** | `dsr.py` | Expected best SR from N trials under null |
| **Standard Error** | `psr.py` | Lo (2002) correction for non-normality |

### Conventions

- **Kurtosis:** Regular kurtosis (normal = 3.0), NOT excess kurtosis.
- **Std denominator:** T-1 (Bessel's correction, sample std).
- **Annualization:** Explicit `periods_per_year` parameter only.
- **Risk-free rate:** Default 0.0 (excess returns = raw returns).
- **Mathematical notation:** T = sample size, N = trial count.

### References

- Bailey, D.H. & López de Prado, M. (2014). "The Deflated Sharpe Ratio." *Journal of Portfolio Management*, 40(5), 94-107.
- Bailey, D.H. & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier." *Journal of Risk*, 15(2), 3-44.
- Lo, A. (2002). "The Statistics of Sharpe Ratios." *Financial Analysts Journal*, 58(4), 36-52.

### Known Limitations

- PSR/DSR assume i.i.d. returns.
- Expected maximum SR is an asymptotic approximation.
- Trial independence is assumed; correlated trials require adjustment.
- Does not detect look-ahead or survivorship bias.

See `docs/statistics.md` for full mathematical documentation.

## Backtest Engine

The backtest module (`src/quant_engine/backtest/`) provides a minimal vectorized backtest engine for strategy evaluation.

### Architecture

| Module | Purpose |
|--------|---------|
| `data.py` | `PriceData` — immutable price sequence with validation |
| `strategy.py` | `Strategy` Protocol — structural typing for strategies |
| `engine.py` | `run_backtest()` — core engine with temporal enforcement |
| `strategies.py` | Built-in strategies: BuyAndHold, AlwaysFlat, AlwaysShort, SMACrossover |
| `types.py` | `BacktestResult`, `OverfittingDiagnosis` — Pydantic result models |

### Temporal Convention (Critical)

```
signal[t] → position[t] → return[t+1]
```

- `signal[t]` uses information through `prices[t]` only.
- `position[t] = signal[t]`, active during period `t+1`.
- `strategy_return[t] = position[t] * asset_return[t]`.

The engine shifts signals by 1 period to enforce this. Strategies cannot introduce look-ahead bias by construction.

### Strategy Protocol

```python
class Strategy(Protocol):
    name: str
    def generate_signal(self, data: PriceData) -> np.ndarray: ...
```

Signals are in {-1, 0, +1}. Structural typing (duck-typing compatible).

### Key Constraints

- Close prices only (Phase 0)
- Zero transaction costs
- No fractional sizing
- Minimum 2 price observations

### Integration with Statistics

`BacktestResult.to_statistics()` converts to `StrategyStatistics` for PSR/DSR evaluation.

See `docs/backtesting.md` for full documentation.

## Strategy Generator

The strategy generator (`src/quant_engine/backtest/generator.py`) produces parameterized trading strategies for overfitting experiments.

### Architecture

| Component | Purpose |
|-----------|---------|
| `StrategySpec` | Serializable specification (family, params, seed, ID) |
| `StrategyFamily` | ABC defining parameter space and strategy construction |
| `StrategyGenerator` | Deterministic generation from master seed |
| `ParameterSpace` | Parameter bounds and validation |

### Strategy Families

| Family | Parameters | Signal Logic |
|--------|-----------|-------------|
| `random_threshold` | threshold (float) | Return vs threshold |
| `random_sma` | fast_window, slow_window (int) | SMA crossover |
| `random_momentum` | lookback (int) | Price change over lookback |
| `random_mean_reversion` | lookback (int), threshold (float) | Deviation from SMA |

### Key Properties

- **Deterministic:** Same seed + same config = same strategies
- **Independent of data:** Generator never accesses PriceData or BacktestResult
- **Serializable:** StrategySpec supports JSON roundtrip
- **Identifiable:** Each strategy has a stable SHA-256-derived ID
- **Inspectable:** All parameters and generation metadata exposed

### RNG Design

```
master_seed → hash(master_seed, index) → per-strategy seed → per-strategy RNG
```

### Constraint

Generator does NOT run backtests. Generation and evaluation are strictly separated.

See `docs/strategy_generation.md` for full documentation.

## Backtest Overfitting Analysis

The overfitting module (`src/quant_engine/overfitting/`) quantifies the statistical danger of selecting the best strategy from many tested strategies.

### Components

| Module | Purpose |
|--------|---------|
| `results.py` | `OverfittingAnalysisResult` — Pydantic model with complete analysis output |
| `analysis.py` | `analyze_backtest_overfitting()` — applies DSR to ExperimentResult |

### Statistical Framework

| Metric | Purpose |
|--------|---------|
| **Expected Max Sharpe** | Baseline: best SR from N random strategies under zero skill |
| **DSR** | Probability that the best strategy's true SR exceeds the baseline |

### Key Design Decisions

1. **Consumes ExperimentResult** — No strategy regeneration or backtest reruns
2. **Reuses existing primitives** — `expected_max_sharpe()`, `dsr_from_stats()` from statistics module
3. **Conservative independence assumption** — Uses nominal trial count N despite correlated strategies
4. **No strategy selection** — Reports on the experiment; does not choose strategies
5. **No PBO/CSCV** — Deferred to future phases

### Integration Flow

```
ExperimentRunner.run()
       ↓
ExperimentResult
       ↓
analyze_backtest_overfitting()
       ↓
OverfittingAnalysisResult
```

See `docs/overfitting.md` for full documentation.

## Research Study

The research module (`src/quant_engine/research/`) orchestrates existing components into reproducible quantitative demonstrations.

### Architecture

```
OverfittingStudyConfig
        ↓
run_overfitting_study()
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
OverfittingStudyResult
```

### Components

| Module | Purpose |
|--------|---------|
| `overfitting_study.py` | `OverfittingStudyConfig`, `OverfittingStudyResult`, `run_overfitting_study()` |

### Design Principles

- **Prefix-based seed policy:** N=10 uses strategies 0..9, N=100 uses 0..99, N=1000 uses 0..999
- **Same price data:** All experiments use identical synthetic data
- **Deterministic:** Same config always produces same results
- **No modification of existing components:** Pure orchestration layer

See `docs/research/overfitting-study.md` for the methodology and results.

## Security Principles

- No API keys in code
- Environment variables for configuration
- Paper trading only (no real execution)
- Data directory is local and gitignored
