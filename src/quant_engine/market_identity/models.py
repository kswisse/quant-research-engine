"""Market identity domain models.

Provider-independent representation of market identity for cross-venue
resolution semantics.  All IDs are deterministic.  All models are frozen.

Architecture:
    EventIdentity      — the underlying event (election, price target, etc.)
    EventBoundary      — temporal boundaries of the event
    OutcomeIdentity    — a specific outcome within the event
    ResolutionRule     — how the outcome is determined
    ResolutionState    — whether the market is resolved, unresolved, or void
    SettlementTerms    — how payouts are calculated
    TemporalScope      — when the market evaluates the outcome
    MarketIdentity     — composite of all the above + provider attribution
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class EventBoundary(BaseModel):
    """Temporal boundaries for an event.

    Attributes:
        close_time: When betting closes (UTC).
        expiry_time: When the market expires (UTC).
    """

    model_config = {"frozen": True}

    close_time: datetime
    expiry_time: datetime


class EventIdentity(BaseModel):
    """Immutable identity for a prediction market event.

    Attributes:
        event_id: Provider-independent event identifier.
        title: Human-readable event title.
        category: Optional event category.
        boundary: Temporal boundaries.
        schema_version: Schema version for forward compatibility.
    """

    model_config = {"frozen": True}

    event_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    category: str = ""
    boundary: EventBoundary
    schema_version: str = "1"


class OutcomeIdentity(BaseModel):
    """Immutable identity for a specific outcome.

    Attributes:
        outcome_id: Provider-independent outcome identifier.
        label: Human-readable outcome label.
        description: Detailed outcome description.
    """

    model_config = {"frozen": True}

    outcome_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    description: str = ""


class ResolutionRule(BaseModel):
    """Immutable resolution rule for a market.

    Attributes:
        source: Resolution data source (e.g. "associated-press").
        method: Resolution method (e.g. "official-certification").
        authority: Resolution authority (e.g. "usc").
        is_definitive: Whether this source provides definitive resolution.
    """

    model_config = {"frozen": True}

    source: str = Field(min_length=1)
    method: str = Field(min_length=1)
    authority: str = Field(min_length=1)
    is_definitive: bool = True


class ResolutionState(StrEnum):
    """Immutable resolution state for a market.

    Represents the current settlement status of a market.
    Two markets with incompatible resolution states cannot be
    economically equivalent for cross-venue arbitrage.

    Attributes:
        RESOLVED: Market has been officially resolved.
        UNRESOLVED: Market is still open / awaiting resolution.
        VOID: Market was cancelled, voided, or declared invalid.
    """

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    VOID = "void"


class SettlementTerms(BaseModel):
    """Immutable settlement terms for a market.

    Attributes:
        payout_type: Type of payout (e.g. "binary", "range").
        payout_cap: Maximum payout per contract.
        settlement_formula: How payouts are calculated.
        settlement_currency: Currency for settlement (e.g. "USD", "USDC").
    """

    model_config = {"frozen": True}

    payout_type: str = Field(min_length=1)
    payout_cap: float = Field(gt=0.0)
    settlement_formula: str = Field(min_length=1)
    settlement_currency: str = Field(default="USD", min_length=1)


class TemporalScope(BaseModel):
    """Immutable temporal scope for market evaluation.

    Attributes:
        evaluation_time: When the market evaluates the outcome (UTC).
        is_point_in_time: True for snapshot evaluation, False for path-dependent.
        start_time: Start of evaluation window (for path-dependent).
    """

    model_config = {"frozen": True}

    evaluation_time: datetime
    is_point_in_time: bool
    start_time: datetime | None = None


class MarketIdentity(BaseModel):
    """Immutable composite market identity.

    Combines event, outcome, resolution, settlement, and provider attribution
    into a single canonical identity.

    Attributes:
        provider: Provider identifier (e.g. "polymarket", "kalshi").
        provider_market_id: Provider's market identifier.
        event: Event identity.
        outcome: Outcome identity.
        resolution: Resolution rule.
        resolution_state: Whether the market is resolved, unresolved, or void.
        settlement: Settlement terms.
        temporal_scope: Temporal evaluation scope.
        schema_version: Schema version for forward compatibility.
    """

    model_config = {"frozen": True}

    provider: str = Field(min_length=1)
    provider_market_id: str = Field(min_length=1)
    event: EventIdentity
    outcome: OutcomeIdentity
    resolution: ResolutionRule
    resolution_state: ResolutionState = ResolutionState.UNRESOLVED
    settlement: SettlementTerms
    temporal_scope: TemporalScope
    schema_version: str = "1"

    @property
    def canonical_identity(self) -> str:
        """Deterministic canonical identity hash.

        Derived from canonical JSON of semantic fields only.
        Provider-specific identifiers are EXCLUDED — two markets with
        identical semantic content from different providers produce
        the same canonical_identity.
        """
        d: dict[str, Any] = {
            "event_id": self.event.event_id,
            "outcome_id": self.outcome.outcome_id,
            "resolution_source": self.resolution.source,
            "resolution_method": self.resolution.method,
            "resolution_authority": self.resolution.authority,
            "resolution_state": self.resolution_state.value,
            "settlement_formula": self.settlement.settlement_formula,
            "settlement_currency": self.settlement.settlement_currency,
            "settlement_payout_cap": self.settlement.payout_cap,
            "settlement_payout_type": self.settlement.payout_type,
            "evaluation_time": self.temporal_scope.evaluation_time.isoformat(),
            "is_point_in_time": self.temporal_scope.is_point_in_time,
        }
        if self.temporal_scope.start_time is not None:
            d["start_time"] = self.temporal_scope.start_time.isoformat()
        canonical = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
