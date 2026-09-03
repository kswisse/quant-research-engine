# Quant Research Engine

A quantitative research platform for backtest overfitting detection, prediction market analysis, options analytics, and information diffusion modeling.

## Status

**Phase 0 — Foundation.** Currently building the Backtest Overfitting Detector (System D).

This is research/paper-trading software only. **No live trading. No real money. No financial advice.**

## Planned Systems

| System | Name | Status |
|--------|------|--------|
| A | Prediction Market Mispricing Engine | Planned |
| B | Hawkes Information Diffusion Model | Planned |
| C | Breeden–Litzenberger Risk-Neutral Distribution | Planned |
| D | Backtest Overfitting Detector | **In Progress** |

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

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=quant_engine

# Run specific test file
pytest tests/unit/test_random.py -v
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

## Architecture

See `docs/architecture.md` for the full architecture documentation.

Key principle: **Core/domain code must NOT depend on concrete database, API, filesystem, or frontend implementations.**

## License

Proprietary — internal research use only.
