"""Tests for experiment configuration."""

from __future__ import annotations

import pytest

from quant_engine.experiments.config import ExperimentConfig


class TestExperimentConfig:
    """Test ExperimentConfig construction and validation."""

    def test_valid_config(self) -> None:
        config = ExperimentConfig(
            seed=42,
            num_strategies=10,
            price_data=[100.0, 101.0, 102.0],
        )
        assert config.seed == 42
        assert config.num_strategies == 10
        assert len(config.price_data) == 3

    def test_rejects_negative_seed(self) -> None:
        with pytest.raises(Exception):
            ExperimentConfig(
                seed=-1,
                num_strategies=10,
                price_data=[100.0, 101.0],
            )

    def test_rejects_zero_strategies(self) -> None:
        with pytest.raises(Exception):
            ExperimentConfig(
                seed=42,
                num_strategies=0,
                price_data=[100.0, 101.0],
            )

    def test_rejects_negative_strategies(self) -> None:
        with pytest.raises(Exception):
            ExperimentConfig(
                seed=42,
                num_strategies=-5,
                price_data=[100.0, 101.0],
            )

    def test_frozen(self) -> None:
        config = ExperimentConfig(
            seed=42, num_strategies=10, price_data=[100.0, 101.0]
        )
        with pytest.raises(Exception):
            config.seed = 99  # type: ignore[misc]


class TestExperimentID:
    """Test deterministic experiment ID derivation."""

    def test_same_config_same_id(self) -> None:
        c1 = ExperimentConfig(seed=42, num_strategies=10, price_data=[100.0, 101.0])
        c2 = ExperimentConfig(seed=42, num_strategies=10, price_data=[100.0, 101.0])
        assert c1.experiment_id == c2.experiment_id

    def test_different_seed_different_id(self) -> None:
        c1 = ExperimentConfig(seed=42, num_strategies=10, price_data=[100.0, 101.0])
        c2 = ExperimentConfig(seed=99, num_strategies=10, price_data=[100.0, 101.0])
        assert c1.experiment_id != c2.experiment_id

    def test_different_num_strategies_different_id(self) -> None:
        c1 = ExperimentConfig(seed=42, num_strategies=10, price_data=[100.0, 101.0])
        c2 = ExperimentConfig(seed=42, num_strategies=20, price_data=[100.0, 101.0])
        assert c1.experiment_id != c2.experiment_id

    def test_different_data_different_id(self) -> None:
        c1 = ExperimentConfig(seed=42, num_strategies=10, price_data=[100.0, 101.0])
        c2 = ExperimentConfig(seed=42, num_strategies=10, price_data=[100.0, 102.0])
        assert c1.experiment_id != c2.experiment_id


class TestSerialization:
    """Test JSON roundtrip serialization."""

    def test_to_dict_roundtrip(self) -> None:
        config = ExperimentConfig(
            seed=42,
            num_strategies=10,
            price_data=[100.0, 101.0, 102.0],
            generator_version="1",
            name="test_experiment",
        )
        d = config.to_dict()
        config2 = ExperimentConfig.from_dict(d)
        assert config == config2

    def test_to_dict_contains_experiment_id(self) -> None:
        config = ExperimentConfig(seed=42, num_strategies=5, price_data=[100.0])
        d = config.to_dict()
        assert "experiment_id" in d
        assert d["experiment_id"] == config.experiment_id
