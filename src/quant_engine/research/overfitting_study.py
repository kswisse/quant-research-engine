"""Research study: overfitting demonstration across strategy populations.

Orchestrates existing components to demonstrate how multiple testing
creates apparently strong strategies from noise.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, Field

from quant_engine.experiments.config import ExperimentConfig
from quant_engine.experiments.runner import ExperimentRunner
from quant_engine.overfitting.analysis import analyze_backtest_overfitting
from quant_engine.overfitting.results import OverfittingAnalysisResult  # noqa: TC001


class OverfittingStudyConfig(BaseModel):
    """Configuration for an overfitting research study.

    Attributes:
        seed: Master seed for deterministic strategy generation.
        strategy_counts: Tuple of strategy counts to test (e.g. (10, 100, 1000)).
        price_data: Closing prices for backtesting.
        generator_version: Version of the strategy generator.
        name: Optional human-readable study name.
    """

    model_config = {"frozen": True}

    seed: int = Field(ge=0)
    strategy_counts: tuple[int, ...]
    price_data: list[float]
    generator_version: str = "1"
    name: str = ""

    def model_post_init(self, __context: Any) -> None:
        """Validate strategy_counts after initialization."""
        if not self.strategy_counts:
            raise ValueError("strategy_counts must not be empty")
        for i, count in enumerate(self.strategy_counts):
            if count < 1:
                raise ValueError(f"strategy_counts[{i}] must be >= 1, got {count}")

    @property
    def study_id(self) -> str:
        """Deterministic study identifier.

        Derived from canonical JSON of seed, strategy_counts, and price_data.
        Same configuration always produces the same ID.
        """
        canonical = json.dumps(
            {
                "seed": self.seed,
                "strategy_counts": list(self.strategy_counts),
                "price_data": self.price_data,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "seed": self.seed,
            "strategy_counts": list(self.strategy_counts),
            "price_data": self.price_data,
            "generator_version": self.generator_version,
            "name": self.name,
            "study_id": self.study_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OverfittingStudyConfig:
        """Deserialize from a dictionary."""
        return cls(
            seed=d["seed"],
            strategy_counts=tuple(d["strategy_counts"]),
            price_data=d["price_data"],
            generator_version=str(d.get("generator_version", "1")),
            name=str(d.get("name", "")),
        )


class OverfittingStudyResult(BaseModel):
    """Complete result of an overfitting research study.

    Aggregates overfitting analyses across multiple strategy-count experiments,
    all using the same price data and seed policy.

    Attributes:
        config: The study configuration.
        analyses: One OverfittingAnalysisResult per strategy count, in order.
    """

    model_config = {"frozen": True}

    config: OverfittingStudyConfig
    analyses: list[OverfittingAnalysisResult]

    @property
    def study_id(self) -> str:
        """Delegated to config.study_id."""
        return self.config.study_id

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "study_id": self.study_id,
            "config": self.config.to_dict(),
            "analyses": [a.to_dict() for a in self.analyses],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OverfittingStudyResult:
        """Deserialize from a dictionary."""
        config = OverfittingStudyConfig.from_dict(d["config"])
        analyses = [OverfittingAnalysisResult(**a) for a in d["analyses"]]
        return cls(config=config, analyses=analyses)


def run_overfitting_study(
    config: OverfittingStudyConfig,
) -> OverfittingStudyResult:
    """Run a complete overfitting research study.

    For each strategy count in config.strategy_counts, runs an experiment
    with that many strategies, then analyzes the result for overfitting.

    All experiments use the same price data and seed, ensuring that
    smaller populations are deterministic subsets of larger ones
    (prefix-based seed policy).

    Args:
        config: Study configuration.

    Returns:
        OverfittingStudyResult with one analysis per strategy count.

    Raises:
        ValueError: If config is invalid.
    """
    runner = ExperimentRunner()
    analyses = []

    for n_strategies in config.strategy_counts:
        exp_config = ExperimentConfig(
            seed=config.seed,
            num_strategies=n_strategies,
            price_data=config.price_data,
            generator_version=config.generator_version,
            name=f"{config.name}_N{n_strategies}" if config.name else "",
        )
        experiment = runner.run(exp_config)
        analysis = analyze_backtest_overfitting(experiment)
        analyses.append(analysis)

    return OverfittingStudyResult(config=config, analyses=analyses)
