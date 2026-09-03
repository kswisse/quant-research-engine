# Strategy Generation

Controlled synthetic strategy generator for backtest-overfitting experiments.

## Purpose

The generator produces parameterized trading strategies from controlled families. This enables experiments like:

```
Generate N strategies → Backtest every strategy → Observe Sharpe distribution → Select maximum → Evaluate selection bias → Apply DSR
```

The key question: *How impressive can the best backtest appear when many strategies are tested, even with no exploitable signal?*

## Architecture

```
StrategySpec         — serializable parameter representation
StrategyFamily       — defines parameter space and strategy construction
StrategyGenerator    — deterministic generation from master seed
Strategy             — executable signal generator (Protocol)
BacktestEngine       — evaluates strategies (separate phase)
```

### Separation of Concerns

| Component | Knows about | Does not know about |
|-----------|------------|-------------------|
| StrategyGenerator | Parameter spaces, RNG | PriceData, BacktestResult |
| StrategyFamily | Parameter bounds | Future prices, Sharpe |
| StrategySpec | Parameters, seed | Market data, performance |
| Strategy | PriceData (at execution) | Future prices, other strategies |

**Critical:** The generator NEVER accesses market data or backtest performance. This prevents hidden optimization.

## Controlled Randomness

### RNG Design

```
master_seed (e.g., 42)
    ↓
hash(master_seed, strategy_index) → per-strategy seed
    ↓
numpy.default_rng(per_strategy_seed) → per-strategy RNG
    ↓
strategy.sample_parameters(rng) → deterministic parameters
```

Each strategy gets its own RNG stream derived from the master seed and its index. Same `(seed, index)` always produces the same strategy.

### Reproducibility

```python
gen1 = StrategyGenerator(seed=42)
gen2 = StrategyGenerator(seed=42)
assert gen1.generate(100) == gen2.generate(100)  # identical
```

## Strategy ID

Deterministic identifier derived from canonical JSON:

```json
{"family":"random_sma","generator_version":"1","parameters":{"fast_window":10,"slow_window":50},"seed":123456}
```

SHA-256 hash, truncated to 16 hex characters. Stable across processes.

## Strategy Families

### Family A: Random Threshold

Signal based on return exceeding a threshold.

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| threshold | float | [0.001, 0.10] | Return threshold |

Signal: `+1` if `r[t] > threshold`, `-1` if `r[t] < -threshold`, else `0`.

### Family B: Random SMA Crossover

Signal based on fast/slow moving average comparison.

| Parameter | Type | Range | Constraint |
|-----------|------|-------|------------|
| fast_window | int | [3, 20] | `fast < slow` |
| slow_window | int | [10, 60] | `fast < slow` |

### Family C: Random Momentum

Signal based on price change over lookback period.

| Parameter | Type | Range |
|-----------|------|-------|
| lookback | int | [5, 60] |

Signal: `+1` if `prices[t]/prices[t-lookback] > 1`, `-1` if `< 1`.

### Family D: Random Mean Reversion

Signal based on deviation from moving average.

| Parameter | Type | Range |
|-----------|------|-------|
| lookback | int | [5, 40] |
| threshold | float | [0.005, 0.05] |

Signal: `+1` if price below SMA by more than threshold (buy), `-1` if above (sell).

## Serialization

StrategySpec is serializable to JSON-compatible dicts:

```python
spec.to_dict()   # → dict
StrategySpec.from_dict(d)  # ← dict
```

Fields: `family`, `parameters`, `seed`, `strategy_index`, `generator_version`, `strategy_id`.

## Look-Ahead Protection

The generator is structurally independent of market data:

- No `PriceData` import in generator module
- No `run_backtest` calls
- No `Sharpe` calculation
- Parameters sampled from fixed distributions, not optimized

## Strategy Correlation

**Warning:** Multiple strategies from the same family are NOT statistically independent, even with different parameters. SMA variants produce correlated signals when windows overlap. This matters when interpreting DSR results.

The generator makes it possible to study correlation later, but does not claim independence.

## Limitations

- **Parameter ranges are fixed.** Changing ranges changes experiments.
- **Simple strategies only.** Complexity is an experimental variable, not a generator feature.
- **No cross-family interaction.** Each strategy belongs to one family.
- **Independent parameters ≠ independent returns.** Strategies in the same family may produce correlated returns.

## Usage

```python
from quant_engine.backtest import StrategyGenerator

gen = StrategyGenerator(seed=42)
specs = gen.generate(1000)

# Each spec is fully inspectable
for spec in specs:
    print(f"{spec.strategy_id}: {spec.family} {spec.parameters}")

# Reconstruct a strategy
strategy = gen.create_strategy(specs[0])
# strategy is ready for run_backtest(data, strategy)
```
