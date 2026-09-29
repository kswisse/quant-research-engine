"""Reproducible random number generation.

All quantitative code must use an explicit RNG instance rather than
global random state. This ensures:
- Same seed + same inputs → same outputs
- No hidden dependence on global state
- Deterministic experiments

Usage:
    from quant_engine.core.random import SeedableRNG

    rng = SeedableRNG(seed=42)
    values = rng.normal(size=100)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator


class SeedableRNG:
    """A seeded random number generator for reproducible quantitative research.

    Wraps numpy.random.Generator with an explicit seed. Two instances
    created with the same seed produce identical sequences when given
    the same call pattern.
    """

    def __init__(self, seed: int) -> None:
        if seed < 0:
            raise ValueError(f"Seed must be non-negative, got {seed}")
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def generator(self) -> Generator:
        """Access the underlying numpy Generator for advanced operations."""
        return self._rng

    def normal(
        self, *, loc: float = 0.0, scale: float = 1.0, size: int | tuple[int, ...] = 1
    ) -> np.ndarray:
        return self._rng.normal(loc=loc, scale=scale, size=size)

    def uniform(
        self, *, low: float = 0.0, high: float = 1.0, size: int | tuple[int, ...] = 1
    ) -> np.ndarray:
        return self._rng.uniform(low=low, high=high, size=size)

    def integers(self, *, low: int, high: int, size: int | tuple[int, ...] = 1) -> np.ndarray:
        return self._rng.integers(low=low, high=high, size=size)

    def choice(self, a: int | list[int], size: int | None = 1, replace: bool = True) -> np.ndarray:
        # numpy returns a scalar when size=None, an ndarray otherwise
        return self._rng.choice(a, size=size, replace=replace)  # type: ignore[return-value]

    def permutation(self, x: int | np.ndarray) -> np.ndarray:
        return self._rng.permutation(x)

    def __repr__(self) -> str:
        return f"SeedableRNG(seed={self._seed})"
