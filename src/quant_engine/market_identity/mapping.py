"""Market identity mapping — cross-venue resolution semantics.

Validates that two MarketIdentity instances refer to the same (or
opposite) underlying market, accounting for resolution source,
expiry, settlement, and temporal scope.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, model_validator

from quant_engine.market_identity.errors import (
    IncompatibleOutcomeError,
    IncompatibleResolutionError,
    IncompatibleResolutionStateError,
    IncompatibleSettlementError,
    MarketIdentityError,
)
from quant_engine.market_identity.models import MarketIdentity, ResolutionState

# Valid relationship types
_VALID_RELATIONSHIPS = frozenset({"same_market", "opposite_outcome"})


class MarketMapping(BaseModel):
    """Immutable mapping between two MarketIdentity instances.

    Attributes:
        source: The source market identity.
        target: The target market identity.
        relationship: The semantic relationship ("same_market" or "opposite_outcome").
        mapping_id: Deterministic mapping identifier.
        schema_version: Schema version for forward compatibility.
    """

    model_config = {"frozen": True}

    source: MarketIdentity
    target: MarketIdentity
    relationship: str
    schema_version: str = "1"

    @model_validator(mode="after")
    def _validate_mapping(self) -> MarketMapping:
        """Validate that the mapping is semantically consistent."""
        # 1. Validate relationship type
        if self.relationship not in _VALID_RELATIONSHIPS:
            raise MarketIdentityError(
                f"Invalid relationship '{self.relationship}'. "
                f"Must be one of: {sorted(_VALID_RELATIONSHIPS)}"
            )

        # 2. For same_market, validate compatibility
        if self.relationship == "same_market":
            self._validate_same_market()

        return self

    def _validate_same_market(self) -> None:
        """Validate that source and target refer to the same market."""
        src = self.source
        tgt = self.target

        # Check event identity
        if src.event.event_id != tgt.event.event_id:
            raise MarketIdentityError(
                f"Event identity mismatch: '{src.event.event_id}' != '{tgt.event.event_id}'"
            )

        # Check outcome identity
        if src.outcome.outcome_id != tgt.outcome.outcome_id:
            raise IncompatibleOutcomeError(
                f"Outcome mismatch: '{src.outcome.outcome_id}' != '{tgt.outcome.outcome_id}'"
            )

        # Check resolution compatibility
        if src.resolution.source != tgt.resolution.source:
            raise IncompatibleResolutionError(
                f"Resolution source mismatch: "
                f"'{src.resolution.source}' != '{tgt.resolution.source}'"
            )
        if src.resolution.method != tgt.resolution.method:
            raise IncompatibleResolutionError(
                f"Resolution method mismatch: "
                f"'{src.resolution.method}' != '{tgt.resolution.method}'"
            )
        if src.resolution.authority != tgt.resolution.authority:
            raise IncompatibleResolutionError(
                f"Resolution authority mismatch: "
                f"'{src.resolution.authority}' != '{tgt.resolution.authority}'"
            )

        # Check resolution state compatibility
        # A void market cannot be equivalent to a resolved/unresolved market
        if (
            src.resolution_state != tgt.resolution_state
            and (
                src.resolution_state == ResolutionState.VOID
                or tgt.resolution_state == ResolutionState.VOID
            )
        ):
            raise IncompatibleResolutionStateError(
                f"Resolution state mismatch: "
                f"'{src.resolution_state.value}' != '{tgt.resolution_state.value}'"
            )

        # Check settlement compatibility
        if src.settlement.settlement_formula != tgt.settlement.settlement_formula:
            raise IncompatibleSettlementError(
                f"Settlement formula mismatch: "
                f"'{src.settlement.settlement_formula}' != '{tgt.settlement.settlement_formula}'"
            )
        if src.settlement.payout_cap != tgt.settlement.payout_cap:
            raise IncompatibleSettlementError(
                f"Payout cap mismatch: {src.settlement.payout_cap} != {tgt.settlement.payout_cap}"
            )
        if src.settlement.payout_type != tgt.settlement.payout_type:
            raise IncompatibleSettlementError(
                f"Payout type mismatch: "
                f"'{src.settlement.payout_type}' != '{tgt.settlement.payout_type}'"
            )
        if src.settlement.settlement_currency != tgt.settlement.settlement_currency:
            raise IncompatibleSettlementError(
                f"Settlement currency mismatch: "
                f"'{src.settlement.settlement_currency}' != '{tgt.settlement.settlement_currency}'"
            )

        # Check temporal scope
        if src.temporal_scope.evaluation_time != tgt.temporal_scope.evaluation_time:
            raise MarketIdentityError(
                f"Evaluation time mismatch: "
                f"{src.temporal_scope.evaluation_time.isoformat()} != "
                f"{tgt.temporal_scope.evaluation_time.isoformat()}"
            )
        if src.temporal_scope.is_point_in_time != tgt.temporal_scope.is_point_in_time:
            raise MarketIdentityError(
                f"Temporal scope mismatch: point_in_time={src.temporal_scope.is_point_in_time} "
                f"!= {tgt.temporal_scope.is_point_in_time}"
            )
        # For path-dependent markets, start_time defines the evaluation window.
        # Different start_time values mean different economic contracts.
        if (
            not src.temporal_scope.is_point_in_time
            and src.temporal_scope.start_time != tgt.temporal_scope.start_time
        ):
            raise MarketIdentityError(
                f"Path-dependent start_time mismatch: "
                f"{src.temporal_scope.start_time} != {tgt.temporal_scope.start_time}"
            )

        # Check event boundary (close_time and expiry_time must match)
        if src.event.boundary.close_time != tgt.event.boundary.close_time:
            raise MarketIdentityError(
                f"Close time mismatch: "
                f"{src.event.boundary.close_time.isoformat()} != "
                f"{tgt.event.boundary.close_time.isoformat()}"
            )
        if src.event.boundary.expiry_time != tgt.event.boundary.expiry_time:
            raise MarketIdentityError(
                f"Expiry time mismatch: "
                f"{src.event.boundary.expiry_time.isoformat()} != "
                f"{tgt.event.boundary.expiry_time.isoformat()}"
            )

    @property
    def mapping_id(self) -> str:
        """Deterministic mapping identifier.

        Derived from canonical JSON of source/target semantic fields
        and relationship type.
        """
        d: dict[str, Any] = {
            "source_canonical": self.source.canonical_identity,
            "target_canonical": self.target.canonical_identity,
            "relationship": self.relationship,
        }
        canonical = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
