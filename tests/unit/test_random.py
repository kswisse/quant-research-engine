"""Tests for reproducible random number generation."""

import numpy as np
import pytest

from quant_engine.core.random import SeedableRNG


class TestSeedableRNG:
    def test_same_seed_same_output(self):
        rng1 = SeedableRNG(seed=42)
        rng2 = SeedableRNG(seed=42)
        values1 = rng1.normal(size=100)
        values2 = rng2.normal(size=100)
        np.testing.assert_array_equal(values1, values2)

    def test_different_seed_different_output(self):
        rng1 = SeedableRNG(seed=42)
        rng2 = SeedableRNG(seed=99)
        values1 = rng1.normal(size=100)
        values2 = rng2.normal(size=100)
        assert not np.array_equal(values1, values2)

    def test_deterministic_across_calls(self):
        rng = SeedableRNG(seed=123)
        a = rng.normal(size=10)
        rng2 = SeedableRNG(seed=123)
        b = rng2.normal(size=10)
        np.testing.assert_array_equal(a, b)

    def test_negative_seed_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            SeedableRNG(seed=-1)

    def test_uniform_distribution(self):
        rng = SeedableRNG(seed=42)
        values = rng.uniform(low=0.0, high=1.0, size=1000)
        assert values.shape == (1000,)
        assert np.all(values >= 0.0)
        assert np.all(values <= 1.0)

    def test_integers_distribution(self):
        rng = SeedableRNG(seed=42)
        values = rng.integers(low=0, high=10, size=1000)
        assert values.shape == (1000,)
        assert np.all(values >= 0)
        assert np.all(values < 10)

    def test_repr(self):
        rng = SeedableRNG(seed=42)
        assert repr(rng) == "SeedableRNG(seed=42)"

    def test_generator_access(self):
        rng = SeedableRNG(seed=42)
        gen = rng.generator
        assert isinstance(gen, np.random.Generator)
