# Quant Research Engine

[![CI](https://github.com/kswisse/quant-research-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/kswisse/quant-research-engine/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A quantitative research platform for backtest overfitting detection, prediction market analysis, options analytics, and information diffusion modeling.

## Status

**System D — Backtest Overfitting Detector — shipped.** Statistics (Sharpe, PSR, DSR), the backtest engine, the strategy generator, the experiment runner, overfitting analysis, and the overfitting research study are all on `master`.

**System A — Prediction Market Mispricing Engine — in progress.** Market data contract, Polymarket and Kalshi read-only adapters, dataset persistence and replay, the order book model with a mechanical execution simulator, and same-market/cross-venue arbitrage detection with cost and market-identity models are shipped — work runs through Phase 2.6 (arbitrage normalization). See `docs/market-data.md`, `docs/order-book.md`, `docs/arbitrage.md`, `docs/cross-venue-arbitrage.md`.

**Systems B (Hawkes Information Diffusion) and C (Breeden–Litzenberger Risk-Neutral Distribution) — not started.**

This is research/paper-trading software only. **No live trading. No real money. No financial advice.**

## Planned Systems

| System | Name | Status |
|--------|------|--------|
| A | Prediction Market Mispricing Engine | **In Progress** — data, order book, and arbitrage stacks shipped (through Phase 2.6) |
| B | Hawkes Information Diffusion Model | Planned |
| C | Breeden–Litzenberger Risk-Neutral Distribution | Planned |
| D | Backtest Overfitting Detector | **Shipped** |

## Development

### Setup

```bash
# Clone and enter the project
cd quant-research-engine

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install in development mode
pip install -e ".[dev]"

# Copy environment config
cp .env.example .env
```

### Linting and Type Checking

```bash
# Format code
ruff format src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## Testing

The suite runs under **pytest** (`testpaths = ["tests"]` in `pyproject.toml`): 34 test files covering unit tests plus Hypothesis property-based tests in `tests/property/`.

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=quant_engine

# Run specific test file
pytest tests/unit/test_random.py -v
```

CI (`.github/workflows/ci.yml`, on every push to `master` and every pull request) is the gate and the source of truth:

| Check | Command |
|-------|---------|
| Lint | `ruff check .` |
| Type check | `mypy` — **strict mode** (`strict = true` in `pyproject.toml`, pinned `mypy==1.20.2` + `scipy-stubs==1.18.1.1`) |
| Tests | `pytest` |

Subsystem design and behavior are documented under [`docs/`](docs/) — see [Documentation](#documentation).

## Architecture

See `docs/architecture.md` for the full architecture documentation.

Key principle: **Core/domain code must NOT depend on concrete database, API, filesystem, or frontend implementations.**

## Documentation

| Topic | Document |
|-------|----------|
| Architecture and current phase | [`docs/architecture.md`](docs/architecture.md) |
| Statistics (Sharpe, PSR, DSR) | [`docs/statistics.md`](docs/statistics.md) |
| Backtest engine | [`docs/backtesting.md`](docs/backtesting.md) |
| Strategy generation | [`docs/strategy_generation.md`](docs/strategy_generation.md) |
| Experiments | [`docs/experiments.md`](docs/experiments.md) |
| Overfitting analysis | [`docs/overfitting.md`](docs/overfitting.md) |
| Overfitting research study | [`docs/research/overfitting-study.md`](docs/research/overfitting-study.md) |
| Market data contract | [`docs/market-data.md`](docs/market-data.md) |
| Polymarket adapter | [`docs/providers/polymarket.md`](docs/providers/polymarket.md) |
| Kalshi adapter | [`docs/providers/kalshi.md`](docs/providers/kalshi.md) |
| Order book and execution | [`docs/order-book.md`](docs/order-book.md) |
| Same-market arbitrage | [`docs/arbitrage.md`](docs/arbitrage.md) |
| Arbitrage costs | [`docs/arbitrage-costs.md`](docs/arbitrage-costs.md) |
| Cross-venue arbitrage | [`docs/cross-venue-arbitrage.md`](docs/cross-venue-arbitrage.md) |
| Market identity | [`docs/market-identity.md`](docs/market-identity.md) |

## License

MIT — see [`LICENSE`](LICENSE).
