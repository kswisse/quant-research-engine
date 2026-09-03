# Backtesting Engine

Minimal vectorized backtest engine for strategy evaluation.

## Temporal Convention (Critical)

The engine enforces a strict temporal convention to prevent look-ahead bias:

```
information at t  →  signal at t  →  position for t+1  →  return at t+1
```

Concretely:
- `signal[t]` is generated using information through `prices[t]` only.
- `position[t] = signal[t]` (active during return period `t+1`).
- `asset_return[t] = prices[t+1] / prices[t] - 1`.
- `strategy_return[t] = position[t] * asset_return[t]`.

The engine shifts signals by 1 period so that `strategy_return[t]` uses `position[t]` (which was set at close of period `t`).

## API

### PriceData

```python
import numpy as np
from quant_engine.backtest import PriceData

data = PriceData(np.array([100.0, 105.0, 110.0, 108.0]))
print(data.n_periods)    # 4
print(data.returns())    # [0.05, 0.0476, -0.0182]
```

Validates: finite, positive, >= 2 observations, 1-D.

### Strategy Protocol

```python
from quant_engine.backtest import Strategy, PriceData
import numpy as np

class MyStrategy:
    name = "my_strategy"

    def generate_signal(self, data: PriceData) -> np.ndarray:
        # signal[t] uses only information through prices[t]
        signals = np.zeros(data.n_periods)
        signals[1:] = 1.0  # long after first observation
        return signals

assert isinstance(MyStrategy(), Strategy)
```

### run_backtest

```python
from quant_engine.backtest import PriceData, BuyAndHold, run_backtest

data = PriceData(np.array([100.0, 105.0, 110.0, 108.0, 113.4]))
result = run_backtest(data, BuyAndHold())

print(result.strategy_returns)  # strategy returns per period
print(result.total_return)       # cumulative return
print(result.positions)          # positions held per return period
```

### Built-in Strategies

| Strategy | Description |
|----------|-------------|
| `BuyAndHold` | Always long (after first observation) |
| `AlwaysFlat` | No exposure |
| `AlwaysShort` | Always short (after first observation) |
| `SMACrossover(fast, slow)` | SMA crossover signals |

## BacktestResult

```python
result = run_backtest(data, BuyAndHold())

# Convert to StrategyStatistics for PSR/DSR
stats = result.to_statistics()  # StrategyStatistics
```

Fields: `strategy_name`, `strategy_parameters`, `strategy_index`, `sharpe_ratio`, `n_observations`, `skewness`, `kurtosis`, `total_return`, `annualized_return`, `max_drawdown`, `is_selected`.

## Output Audit

Every `BacktestOutput` contains:
- `prices`: full price history
- `signals`: strategy signals (full length)
- `positions`: shifted signals (for return periods)
- `asset_returns`: simple asset returns
- `strategy_returns`: position * asset_return
- `transaction_costs`: cost model (zero in Phase 0)

## Constraints

- No look-ahead bias by construction (signals shifted by 1)
- No fractional sizing (signals in {-1, 0, +1})
- Zero transaction costs (Phase 0)
- Close prices only
