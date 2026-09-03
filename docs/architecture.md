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

## Security Principles

- No API keys in code
- Environment variables for configuration
- Paper trading only (no real execution)
- Data directory is local and gitignored
