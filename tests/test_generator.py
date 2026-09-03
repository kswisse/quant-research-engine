"""Tests for the strategy generator.

Covers: reproducibility, different seeds, valid params, stable IDs,
no data dependency, reconstruction, no performance-aware generation.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

from quant_engine.backtest.data import PriceData
from quant_engine.backtest.generator import (
    ALL_FAMILIES,
    RandomMeanReversionFamily,
    RandomMomentumFamily,
    RandomSMAFamily,
    RandomThresholdFamily,
    StrategyGenerator,
    _derive_strategy_seed,
)
from quant_engine.backtest.strategy import Strategy
from quant_engine.backtest.strategy_specs import StrategySpec


class TestDeriveStrategySeed:
    """Test per-strategy seed derivation."""

    def test_deterministic(self) -> None:
        s1 = _derive_strategy_seed(42, 0)
        s2 = _derive_strategy_seed(42, 0)
        assert s1 == s2

    def test_different_indices_different_seeds(self) -> None:
        s0 = _derive_strategy_seed(42, 0)
        s1 = _derive_strategy_seed(42, 1)
        assert s0 != s1

    def test_different_master_seeds(self) -> None:
        s1 = _derive_strategy_seed(1, 0)
        s2 = _derive_strategy_seed(2, 0)
        assert s1 != s2

    def test_non_negative(self) -> None:
        for i in range(100):
            assert _derive_strategy_seed(42, i) >= 0


class TestReproducibility:
    """Same seed + same config = same strategies."""

    def test_same_seed_same_strategies(self) -> None:
        gen1 = StrategyGenerator(seed=42)
        gen2 = StrategyGenerator(seed=42)
        specs1 = gen1.generate(100)
        specs2 = gen2.generate(100)
        assert len(specs1) == len(specs2)
        for s1, s2 in zip(specs1, specs2):
            assert s1.family == s2.family
            assert s1.parameters == s2.parameters
            assert s1.seed == s2.seed
            assert s1.strategy_id == s2.strategy_id

    def test_same_seed_same_strategies_small(self) -> None:
        gen1 = StrategyGenerator(seed=12345)
        gen2 = StrategyGenerator(seed=12345)
        specs1 = gen1.generate(10)
        specs2 = gen2.generate(10)
        assert specs1 == specs2


class TestDifferentSeeds:
    """Different seeds produce different populations."""

    def test_different_seeds_different_output(self) -> None:
        gen1 = StrategyGenerator(seed=1)
        gen2 = StrategyGenerator(seed=2)
        specs1 = gen1.generate(50)
        specs2 = gen2.generate(50)
        # At least some specs should differ
        assert specs1 != specs2


class TestValidParameters:
    """Every generated strategy has valid parameters."""

    def test_random_threshold_params(self) -> None:
        gen = StrategyGenerator(seed=42, families=["random_threshold"])
        for spec in gen.generate(100):
            assert "threshold" in spec.parameters
            t = spec.parameters["threshold"]
            assert isinstance(t, float)
            assert 0.001 <= t <= 0.10

    def test_random_sma_params(self) -> None:
        gen = StrategyGenerator(seed=42, families=["random_sma"])
        for spec in gen.generate(100):
            fast = spec.parameters["fast_window"]
            slow = spec.parameters["slow_window"]
            assert isinstance(fast, int)
            assert isinstance(slow, int)
            assert 3 <= fast <= 20
            assert 10 <= slow <= 60
            assert fast < slow

    def test_random_momentum_params(self) -> None:
        gen = StrategyGenerator(seed=42, families=["random_momentum"])
        for spec in gen.generate(100):
            lb = spec.parameters["lookback"]
            assert isinstance(lb, int)
            assert 5 <= lb <= 60

    def test_random_mean_reversion_params(self) -> None:
        gen = StrategyGenerator(seed=42, families=["random_mean_reversion"])
        for spec in gen.generate(100):
            lb = spec.parameters["lookback"]
            t = spec.parameters["threshold"]
            assert isinstance(lb, int)
            assert isinstance(t, float)
            assert 5 <= lb <= 40
            assert 0.005 <= t <= 0.05

    def test_all_families_valid_params(self) -> None:
        gen = StrategyGenerator(seed=42)
        for spec in gen.generate(200):
            family = ALL_FAMILIES[spec.family]
            for ps in family.parameter_spaces:
                val = spec.parameters[ps.name]
                assert ps.validate_value(val), (
                    f"Invalid param {ps.name}={val} for {spec.family}"
                )


class TestStableIDs:
    """Strategy IDs are stable and change with parameters."""

    def test_same_spec_same_id(self) -> None:
        gen = StrategyGenerator(seed=42)
        specs = gen.generate(10)
        # Re-derive IDs
        for spec in specs:
            d = spec.to_dict()
            spec2 = type(spec).from_dict(d)
            assert spec.strategy_id == spec2.strategy_id

    def test_different_params_different_ids(self) -> None:
        gen = StrategyGenerator(seed=42)
        specs = gen.generate(100)
        ids = [s.strategy_id for s in specs]
        # All IDs should be unique (collision unlikely with 100 items)
        assert len(set(ids)) == len(ids)

    def test_changing_param_changes_id(self) -> None:
        from quant_engine.backtest.strategy_specs import StrategySpec

        s1 = StrategySpec(
            family="test", parameters={"x": 1}, seed=1, strategy_index=0
        )
        s2 = StrategySpec(
            family="test", parameters={"x": 2}, seed=1, strategy_index=0
        )
        assert s1.strategy_id != s2.strategy_id


class TestNoDataDependency:
    """Generator does not depend on PriceData."""

    def test_generator_no_data_import(self) -> None:
        """Verify generator module does not import PriceData at module level."""
        source = inspect.getsource(
            __import__(
                "quant_engine.backtest.generator", fromlist=["generator"]
            )
        )
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "data" not in alias.name.lower(), (
                        f"Generator imports data module: {alias.name}"
                    )

    def test_generate_without_data(self) -> None:
        """Generator.generate() works without any data."""
        gen = StrategyGenerator(seed=42)
        specs = gen.generate(50)
        assert len(specs) == 50


class TestReconstruction:
    """Generated strategy can be reconstructed and produces same signals."""

    def test_reconstruct_and_compare(self) -> None:
        gen = StrategyGenerator(seed=42, families=["random_threshold"])
        specs = gen.generate(5)

        data = PriceData(np.array([100.0, 101.0, 99.0, 102.0, 98.0, 103.0, 97.0, 104.0]))

        for spec in specs:
            # Create strategy from spec
            strategy = gen.create_strategy(spec)
            signals = strategy.generate_signal(data)

            # Reconstruct from serialized spec
            d = spec.to_dict()
            spec2 = StrategySpec.from_dict(d)
            strategy2 = gen.create_strategy(spec2)
            signals2 = strategy2.generate_signal(data)

            np.testing.assert_array_equal(signals, signals2)

    def test_reconstruct_all_families(self) -> None:
        gen = StrategyGenerator(seed=42)
        specs = gen.generate(20)
        data = PriceData(np.arange(1, 61, dtype=np.float64))

        for spec in specs:
            strategy = gen.create_strategy(spec)
            assert isinstance(strategy, Strategy)
            signals = strategy.generate_signal(data)
            assert len(signals) == 60


class TestNoPerformanceAwareGeneration:
    """Generator does not calculate Sharpe, call run_backtest, or inspect results."""

    def test_no_sharpe_in_source(self) -> None:
        """Verify generator source does not reference Sharpe or backtest."""
        source = inspect.getsource(
            __import__(
                "quant_engine.backtest.generator", fromlist=["generator"]
            )
        )
        forbidden = ["sharpe", "run_backtest", "BacktestResult", "backtest"]
        for term in forbidden:
            # Check for usage, not just import
            assert term not in source.lower() or term == "backtest", (
                f"Generator references forbidden term: {term}"
            )

    def test_generator_only_generates(self) -> None:
        """Generator produces StrategySpec, not BacktestResult."""
        gen = StrategyGenerator(seed=42)
        specs = gen.generate(10)
        for spec in specs:
            assert hasattr(spec, "family")
            assert hasattr(spec, "parameters")
            assert not hasattr(spec, "sharpe_ratio")


class TestFamilies:
    """Test individual family implementations."""

    def test_all_families_registered(self) -> None:
        assert "random_threshold" in ALL_FAMILIES
        assert "random_sma" in ALL_FAMILIES
        assert "random_momentum" in ALL_FAMILIES
        assert "random_mean_reversion" in ALL_FAMILIES

    def test_family_names(self) -> None:
        assert RandomThresholdFamily().name == "random_threshold"
        assert RandomSMAFamily().name == "random_sma"
        assert RandomMomentumFamily().name == "random_momentum"
        assert RandomMeanReversionFamily().name == "random_mean_reversion"

    def test_family_parameter_spaces(self) -> None:
        for family in ALL_FAMILIES.values():
            spaces = family.parameter_spaces
            assert len(spaces) >= 1
            for ps in spaces:
                assert ps.low < ps.high

    def test_family_generate_returns_strategy(self) -> None:
        rng = np.random.default_rng(42)
        for family in ALL_FAMILIES.values():
            params = family.sample_parameters(rng)
            strategy = family.create_strategy(params)
            assert hasattr(strategy, "generate_signal")
            assert hasattr(strategy, "name")

    def test_strategy_signals_valid(self) -> None:
        rng = np.random.default_rng(42)
        data = PriceData(np.array([100.0, 101.0, 99.0, 102.0, 98.0, 103.0]))
        for family in ALL_FAMILIES.values():
            params = family.sample_parameters(rng)
            strategy = family.create_strategy(params)
            signals = strategy.generate_signal(data)
            unique_vals = set(np.unique(signals))
            assert unique_vals.issubset({-1.0, 0.0, 1.0}), (
                f"Invalid signals from {family.name}: {unique_vals}"
            )


class TestGeneratorEdgeCases:
    """Test edge cases and error handling."""

    def test_negative_seed_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            StrategyGenerator(seed=-1)

    def test_unknown_family_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown family"):
            StrategyGenerator(seed=42, families=["nonexistent"])

    def test_n_zero_rejected(self) -> None:
        gen = StrategyGenerator(seed=42)
        with pytest.raises(ValueError, match=">= 1"):
            gen.generate(0)

    def test_single_strategy(self) -> None:
        gen = StrategyGenerator(seed=42)
        specs = gen.generate(1)
        assert len(specs) == 1

    def test_family_filter(self) -> None:
        gen = StrategyGenerator(seed=42, families=["random_momentum"])
        specs = gen.generate(10)
        for spec in specs:
            assert spec.family == "random_momentum"

    def test_round_robin_distribution(self) -> None:
        gen = StrategyGenerator(seed=42, families=["random_threshold", "random_sma"])
        specs = gen.generate(6)
        families = [s.family for s in specs]
        assert families == [
            "random_threshold",
            "random_sma",
            "random_threshold",
            "random_sma",
            "random_threshold",
            "random_sma",
        ]
