"""Controlled synthetic strategy generator.

This module generates parameterized trading strategies for overfitting experiments.

Architecture:
    StrategyGenerator
        → StrategySpec (parameter representation)
        → Strategy (executable, via strategy factory)
        → BacktestEngine (separate, later)

Key design principles:
    1. Generator does NOT access PriceData or BacktestResult
    2. Each strategy gets its own deterministic RNG stream
    3. Same seed + same config = same strategies (reproducible)
    4. StrategySpec is serializable and has a stable strategy_id

RNG design:
    master_seed
        → hash(master_seed, strategy_index) → per-strategy seed
        → numpy.default_rng(per_strategy_seed) → per-strategy RNG

This ensures independence between strategy generations.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from quant_engine.backtest.strategy_specs import ParameterSpace, StrategySpec

if TYPE_CHECKING:
    from quant_engine.backtest.data import PriceData
    from quant_engine.backtest.strategy import Strategy


class StrategyFamily(ABC):
    """Abstract base class for strategy families.

    A strategy family defines:
    - How to sample parameters from a space
    - How to create a Strategy from parameters
    - What parameter constraints exist
    """

    name: str

    @property
    @abstractmethod
    def parameter_spaces(self) -> list[ParameterSpace]:
        """Define the parameter search space for this family."""
        ...

    @abstractmethod
    def sample_parameters(self, rng: np.random.Generator) -> dict[str, float | int]:
        """Sample random parameters from the family's parameter space.

        Args:
            rng: Numpy random generator (deterministic, per-strategy).

        Returns:
            Dictionary of parameter name to sampled value.
        """
        ...

    @abstractmethod
    def create_strategy(self, parameters: dict[str, float | int]) -> Strategy:
        """Create an executable Strategy from parameters.

        Args:
            parameters: Sampled parameters.

        Returns:
            Strategy instance ready for backtesting.
        """
        ...


def _derive_strategy_seed(master_seed: int, strategy_index: int) -> int:
    """Derive a deterministic per-strategy seed from master seed and index.

    Uses SHA-256 hash to ensure complete independence between seeds.
    Same (master_seed, strategy_index) always produces the same seed.
    """
    data = f"{master_seed}:{strategy_index}".encode()
    h = hashlib.sha256(data).hexdigest()
    return int(h[:8], 16) % (2**31)


class RandomThresholdFamily(StrategyFamily):
    """Family A: Random threshold on returns.

    Signal:
        r[t] = prices[t] / prices[t-1] - 1
        signal[t] = +1 if r[t] > threshold
        signal[t] = -1 if r[t] < -threshold
        signal[t] =  0 otherwise

    Parameters:
        threshold: float in [0.001, 0.10]
    """

    name: str = "random_threshold"

    @property
    def parameter_spaces(self) -> list[ParameterSpace]:
        return [
            ParameterSpace(
                name="threshold",
                param_type="float",
                low=0.001,
                high=0.10,
                description="Return threshold for signal generation",
            )
        ]

    def sample_parameters(self, rng: np.random.Generator) -> dict[str, float | int]:
        threshold = float(rng.uniform(0.001, 0.10))
        return {"threshold": round(threshold, 6)}

    def create_strategy(self, parameters: dict[str, float | int]) -> Strategy:
        threshold = float(parameters["threshold"])

        class _Strategy:
            name = "random_threshold"

            def generate_signal(self, data: PriceData) -> np.ndarray:
                prices = data.prices
                n = len(prices)
                signals = np.zeros(n, dtype=np.float64)
                returns = np.diff(prices) / prices[:-1]
                signals[1:] = np.where(
                    returns > threshold, 1.0, np.where(returns < -threshold, -1.0, 0.0)
                )
                return signals

        return _Strategy()


class RandomSMAFamily(StrategyFamily):
    """Family B: Random moving average crossover.

    Signal:
        fast_sma[t] = mean(prices[t-fast+1:t+1])
        slow_sma[t] = mean(prices[t-slow+1:t+1])
        signal[t] = +1 if fast_sma > slow_sma
        signal[t] = -1 if fast_sma < slow_sma
        signal[t] =  0 otherwise
        (only computed for t >= slow_window)

    Parameters:
        fast_window: int in [3, 20]
        slow_window: int in [10, 60]
        Constraint: fast_window < slow_window
    """

    name: str = "random_sma"

    @property
    def parameter_spaces(self) -> list[ParameterSpace]:
        return [
            ParameterSpace(
                name="fast_window",
                param_type="int",
                low=3,
                high=20,
                description="Fast SMA window",
            ),
            ParameterSpace(
                name="slow_window",
                param_type="int",
                low=10,
                high=60,
                description="Slow SMA window",
            ),
        ]

    def sample_parameters(self, rng: np.random.Generator) -> dict[str, float | int]:
        fast = int(rng.integers(3, 21))
        slow = int(rng.integers(10, 61))
        # Ensure fast < slow
        if fast >= slow:
            slow = fast + 5
            if slow > 60:
                slow = 60
                fast = slow - 5
        return {"fast_window": fast, "slow_window": slow}

    def create_strategy(self, parameters: dict[str, float | int]) -> Strategy:
        fast_window = int(parameters["fast_window"])
        slow_window = int(parameters["slow_window"])

        class _Strategy:
            name = "random_sma"

            def generate_signal(self, data: PriceData) -> np.ndarray:
                prices = data.prices
                n = len(prices)
                signals = np.zeros(n, dtype=np.float64)
                for t in range(slow_window, n):
                    window = prices[: t + 1]
                    fast_sma = float(np.mean(window[-fast_window:]))
                    slow_sma = float(np.mean(window[-slow_window:]))
                    if fast_sma > slow_sma:
                        signals[t] = 1.0
                    elif fast_sma < slow_sma:
                        signals[t] = -1.0
                return signals

        return _Strategy()


class RandomMomentumFamily(StrategyFamily):
    """Family C: Random momentum lookback.

    Signal:
        momentum = prices[t] / prices[t-lookback] - 1
        signal[t] = +1 if momentum > 0
        signal[t] = -1 if momentum < 0
        signal[t] =  0 if momentum == 0
        (only computed for t >= lookback)

    Parameters:
        lookback: int in [5, 60]
    """

    name: str = "random_momentum"

    @property
    def parameter_spaces(self) -> list[ParameterSpace]:
        return [
            ParameterSpace(
                name="lookback",
                param_type="int",
                low=5,
                high=60,
                description="Lookback period for momentum calculation",
            )
        ]

    def sample_parameters(self, rng: np.random.Generator) -> dict[str, float | int]:
        lookback = int(rng.integers(5, 61))
        return {"lookback": lookback}

    def create_strategy(self, parameters: dict[str, float | int]) -> Strategy:
        lookback = int(parameters["lookback"])

        class _Strategy:
            name = "random_momentum"

            def generate_signal(self, data: PriceData) -> np.ndarray:
                prices = data.prices
                n = len(prices)
                signals = np.zeros(n, dtype=np.float64)
                for t in range(lookback, n):
                    momentum = prices[t] / prices[t - lookback] - 1.0
                    if momentum > 0:
                        signals[t] = 1.0
                    elif momentum < 0:
                        signals[t] = -1.0
                return signals

        return _Strategy()


class RandomMeanReversionFamily(StrategyFamily):
    """Family D: Random mean reversion.

    Signal:
        sma = mean(prices[t-lookback+1:t+1])
        deviation = (prices[t] - sma) / sma
        signal[t] = +1 if deviation < -threshold  (underpriced → buy)
        signal[t] = -1 if deviation > threshold   (overpriced → sell)
        signal[t] =  0 otherwise
        (only computed for t >= lookback)

    Parameters:
        lookback: int in [5, 40]
        threshold: float in [0.005, 0.05]
    """

    name: str = "random_mean_reversion"

    @property
    def parameter_spaces(self) -> list[ParameterSpace]:
        return [
            ParameterSpace(
                name="lookback",
                param_type="int",
                low=5,
                high=40,
                description="Lookback period for SMA calculation",
            ),
            ParameterSpace(
                name="threshold",
                param_type="float",
                low=0.005,
                high=0.05,
                description="Deviation threshold for signal generation",
            ),
        ]

    def sample_parameters(self, rng: np.random.Generator) -> dict[str, float | int]:
        lookback = int(rng.integers(5, 41))
        threshold = float(rng.uniform(0.005, 0.05))
        return {"lookback": lookback, "threshold": round(threshold, 6)}

    def create_strategy(self, parameters: dict[str, float | int]) -> Strategy:
        lookback = int(parameters["lookback"])
        threshold = float(parameters["threshold"])

        class _Strategy:
            name = "random_mean_reversion"

            def generate_signal(self, data: PriceData) -> np.ndarray:
                prices = data.prices
                n = len(prices)
                signals = np.zeros(n, dtype=np.float64)
                for t in range(lookback, n):
                    window = prices[t - lookback + 1 : t + 1]
                    sma = float(np.mean(window))
                    deviation = (prices[t] - sma) / sma
                    if deviation < -threshold:
                        signals[t] = 1.0
                    elif deviation > threshold:
                        signals[t] = -1.0
                return signals

        return _Strategy()


# Registry of all families
ALL_FAMILIES: dict[str, StrategyFamily] = {
    "random_threshold": RandomThresholdFamily(),
    "random_sma": RandomSMAFamily(),
    "random_momentum": RandomMomentumFamily(),
    "random_mean_reversion": RandomMeanReversionFamily(),
}


class StrategyGenerator:
    """Deterministic generator of synthetic trading strategies.

    Given a master seed and generation configuration, produces
    reproducible strategy specifications.

    Usage:
        gen = StrategyGenerator(seed=42)
        specs = gen.generate(n=100)

        # Same config → same strategies
        gen2 = StrategyGenerator(seed=42)
        specs2 = gen2.generate(n=100)
        assert specs == specs2
    """

    def __init__(
        self,
        seed: int,
        families: list[str] | None = None,
        generator_version: str = "1",
    ) -> None:
        """Initialize the generator.

        Args:
            seed: Master seed for reproducibility.
            families: List of family names to generate from.
                      If None, uses all families.
            generator_version: Version string for tracking changes.
        """
        if seed < 0:
            raise ValueError(f"Seed must be non-negative, got {seed}")

        self._seed = seed
        self._generator_version = generator_version

        if families is None:
            self._families = list(ALL_FAMILIES.keys())
        else:
            for f in families:
                if f not in ALL_FAMILIES:
                    available = list(ALL_FAMILIES.keys())
                    raise ValueError(
                        f"Unknown family: {f!r}. Available: {available}"
                    )
            self._families = list(families)

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def families(self) -> list[str]:
        return list(self._families)

    @property
    def generator_version(self) -> str:
        return self._generator_version

    def generate(self, n: int) -> list[StrategySpec]:
        """Generate N strategy specifications.

        Strategies are distributed round-robin across families.

        Args:
            n: Number of strategies to generate (must be >= 1).

        Returns:
            List of StrategySpec, one per generated strategy.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        specs = []
        for i in range(n):
            family_name = self._families[i % len(self._families)]
            family = ALL_FAMILIES[family_name]

            # Derive per-strategy seed
            strategy_seed = _derive_strategy_seed(self._seed, i)

            # Create per-strategy RNG
            rng = np.random.default_rng(strategy_seed)

            # Sample parameters
            parameters = family.sample_parameters(rng)

            spec = StrategySpec(
                family=family_name,
                parameters=parameters,
                seed=strategy_seed,
                strategy_index=i,
                generator_version=self._generator_version,
            )
            specs.append(spec)

        return specs

    def create_strategy(self, spec: StrategySpec) -> Strategy:
        """Create an executable Strategy from a specification.

        This does NOT access PriceData or run backtests.
        It only constructs the strategy object.

        Args:
            spec: Strategy specification.

        Returns:
            Strategy instance.
        """
        family = ALL_FAMILIES.get(spec.family)
        if family is None:
            raise ValueError(f"Unknown family: {spec.family!r}")
        return family.create_strategy(spec.parameters)
