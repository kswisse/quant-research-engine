"""Strategy specification and parameter models.

Design decision: StrategySpec is the serializable representation of a strategy.
It contains all information needed to reconstruct and identify a strategy,
without depending on PriceData or any runtime state.

Strategy ID derivation:
    ID = SHA-256 canonical JSON of {family, parameters, seed, generator_version}
    truncated to 16 hex characters for readability.

Canonical serialization:
    - Keys sorted alphabetically
    - No whitespace
    - UTF-8 encoding
    - Parameters use JSON-compatible types only

This ensures stable IDs across processes and platforms.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, Field


class StrategySpec(BaseModel):
    """Complete specification of a generated strategy.

    Attributes:
        family: Strategy family name (e.g., "random_threshold").
        parameters: Family-specific parameters (e.g., {"threshold": 0.01}).
        seed: Deterministic seed used to generate this specific strategy.
        strategy_index: Position in the generation sequence (0-indexed).
        generator_version: Version of the generator used.
    """

    model_config = {"frozen": True}

    family: str
    parameters: dict[str, float | int]
    seed: int = Field(ge=0)
    strategy_index: int = Field(ge=0)
    generator_version: str = "1"

    @property
    def strategy_id(self) -> str:
        """Deterministic strategy identifier.

        Derived from canonical JSON of family, parameters, seed, and
        generator_version. Stable across runs and platforms.
        """
        canonical = self._canonical_json()
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def _canonical_json(self) -> str:
        """Canonical JSON serialization for ID derivation.

        Keys sorted alphabetically, no whitespace, UTF-8.
        """
        d = {
            "family": self.family,
            "parameters": self.parameters,
            "seed": self.seed,
            "generator_version": self.generator_version,
        }
        return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "family": self.family,
            "parameters": self.parameters,
            "seed": self.seed,
            "strategy_index": self.strategy_index,
            "generator_version": self.generator_version,
            "strategy_id": self.strategy_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StrategySpec:
        """Deserialize from a dictionary."""
        return cls(
            family=d["family"],
            parameters=d["parameters"],
            seed=d["seed"],
            strategy_index=d["strategy_index"],
            generator_version=d.get("generator_version", "1"),
        )

    def __repr__(self) -> str:
        return (
            f"StrategySpec(family={self.family!r}, "
            f"id={self.strategy_id}, index={self.strategy_index})"
        )


class ParameterSpace(BaseModel):
    """Definition of a parameter search space for a strategy family.

    Each parameter is defined by its name, type, and bounds.
    Used by the generator to sample parameters.
    """

    model_config = {"frozen": True}

    name: str
    param_type: str  # "int" or "float"
    low: float
    high: float
    description: str = ""

    def validate_value(self, value: float | int) -> bool:
        """Check if a value is within bounds."""
        if self.param_type == "int":
            if not isinstance(value, int) or isinstance(value, bool):
                return False
            return self.low <= value <= self.high
        # float
        return self.low <= float(value) <= self.high
