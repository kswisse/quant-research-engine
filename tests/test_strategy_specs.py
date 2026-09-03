"""Tests for strategy specification and parameter models."""

from __future__ import annotations

import json

import pytest

from quant_engine.backtest.strategy_specs import ParameterSpace, StrategySpec


class TestStrategySpec:
    """Test StrategySpec construction, serialization, and ID derivation."""

    def test_valid_creation(self) -> None:
        spec = StrategySpec(
            family="random_sma",
            parameters={"fast_window": 5, "slow_window": 20},
            seed=42,
            strategy_index=0,
        )
        assert spec.family == "random_sma"
        assert spec.seed == 42
        assert spec.strategy_index == 0

    def test_strategy_id_is_deterministic(self) -> None:
        spec1 = StrategySpec(
            family="random_sma",
            parameters={"fast_window": 5, "slow_window": 20},
            seed=42,
            strategy_index=0,
        )
        spec2 = StrategySpec(
            family="random_sma",
            parameters={"fast_window": 5, "slow_window": 20},
            seed=42,
            strategy_index=0,
        )
        assert spec1.strategy_id == spec2.strategy_id

    def test_different_parameters_different_id(self) -> None:
        spec1 = StrategySpec(
            family="random_sma",
            parameters={"fast_window": 5, "slow_window": 20},
            seed=42,
            strategy_index=0,
        )
        spec2 = StrategySpec(
            family="random_sma",
            parameters={"fast_window": 10, "slow_window": 20},
            seed=42,
            strategy_index=0,
        )
        assert spec1.strategy_id != spec2.strategy_id

    def test_different_family_different_id(self) -> None:
        spec1 = StrategySpec(
            family="random_sma",
            parameters={"fast_window": 5, "slow_window": 20},
            seed=42,
            strategy_index=0,
        )
        spec2 = StrategySpec(
            family="random_threshold",
            parameters={"threshold": 0.01},
            seed=42,
            strategy_index=0,
        )
        assert spec1.strategy_id != spec2.strategy_id

    def test_strategy_id_is_stable_across_serialization(self) -> None:
        spec = StrategySpec(
            family="random_momentum",
            parameters={"lookback": 10},
            seed=99,
            strategy_index=5,
        )
        spec_id = spec.strategy_id
        # Serialize and deserialize
        d = spec.to_dict()
        spec2 = StrategySpec.from_dict(d)
        assert spec2.strategy_id == spec_id

    def test_to_dict(self) -> None:
        spec = StrategySpec(
            family="test",
            parameters={"a": 1, "b": 2.5},
            seed=10,
            strategy_index=3,
            generator_version="2",
        )
        d = spec.to_dict()
        assert d["family"] == "test"
        assert d["parameters"] == {"a": 1, "b": 2.5}
        assert d["seed"] == 10
        assert d["strategy_index"] == 3
        assert d["generator_version"] == "2"
        assert "strategy_id" in d

    def test_from_dict_roundtrip(self) -> None:
        spec = StrategySpec(
            family="random_mean_reversion",
            parameters={"lookback": 15, "threshold": 0.02},
            seed=77,
            strategy_index=42,
        )
        d = spec.to_dict()
        spec2 = StrategySpec.from_dict(d)
        assert spec == spec2

    def test_frozen(self) -> None:
        spec = StrategySpec(
            family="test",
            parameters={"x": 1},
            seed=1,
            strategy_index=0,
        )
        with pytest.raises(Exception):
            spec.family = "other"  # type: ignore[misc]

    def test_repr(self) -> None:
        spec = StrategySpec(
            family="random_sma",
            parameters={"fast_window": 5},
            seed=42,
            strategy_index=0,
        )
        r = repr(spec)
        assert "random_sma" in r
        assert "strategy_id" in r or spec.strategy_id in r


class TestParameterSpace:
    """Test ParameterSpace validation."""

    def test_int_in_range(self) -> None:
        ps = ParameterSpace(name="lookback", param_type="int", low=5, high=60)
        assert ps.validate_value(10)
        assert ps.validate_value(5)
        assert ps.validate_value(60)

    def test_int_out_of_range(self) -> None:
        ps = ParameterSpace(name="lookback", param_type="int", low=5, high=60)
        assert not ps.validate_value(4)
        assert not ps.validate_value(61)

    def test_float_in_range(self) -> None:
        ps = ParameterSpace(name="threshold", param_type="float", low=0.0, high=1.0)
        assert ps.validate_value(0.5)
        assert ps.validate_value(0.0)
        assert ps.validate_value(1.0)

    def test_float_out_of_range(self) -> None:
        ps = ParameterSpace(name="threshold", param_type="float", low=0.0, high=1.0)
        assert not ps.validate_value(-0.1)
        assert not ps.validate_value(1.1)

    def test_int_type_rejects_float(self) -> None:
        ps = ParameterSpace(name="x", param_type="int", low=1, high=10)
        assert not ps.validate_value(3.5)

    def test_int_type_rejects_bool(self) -> None:
        ps = ParameterSpace(name="x", param_type="int", low=1, high=10)
        assert not ps.validate_value(True)
