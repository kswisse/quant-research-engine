"""Experiment configuration.

Immutable, validated configuration for a reproducible experiment.
The config captures everything needed to exactly reproduce an experiment run.

Note: This is distinct from statistics.types.ExperimentConfig which is
used for statistical experiment design. This config is for the
strategy-generation → backtest → evaluation pipeline.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    """Configuration for a single reproducible experiment.

    Attributes:
        seed: Master seed for strategy generation reproducibility.
        num_strategies: Number of strategies to generate and evaluate.
        price_data: List of closing prices for backtesting.
        generator_version: Version of the strategy generator used.
        name: Optional human-readable experiment name.
    """

    model_config = {"frozen": True}

    seed: int = Field(ge=0)
    num_strategies: int = Field(ge=1)
    price_data: list[float]
    generator_version: str = "1"
    name: str = ""

    @property
    def experiment_id(self) -> str:
        """Deterministic experiment identifier.

        Derived from canonical JSON of all configuration fields.
        Same config always produces the same ID.
        """
        canonical = self._canonical_json()
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def _canonical_json(self) -> str:
        """Canonical JSON serialization for ID derivation."""
        d = {
            "seed": self.seed,
            "num_strategies": self.num_strategies,
            "price_data": self.price_data,
            "generator_version": self.generator_version,
            "name": self.name,
        }
        return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "seed": self.seed,
            "num_strategies": self.num_strategies,
            "price_data": self.price_data,
            "generator_version": self.generator_version,
            "name": self.name,
            "experiment_id": self.experiment_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentConfig:
        """Deserialize from a dictionary."""
        return cls(
            seed=d["seed"],
            num_strategies=d["num_strategies"],
            price_data=d["price_data"],
            generator_version=str(d.get("generator_version", "1")),
            name=str(d.get("name", "")),
        )
