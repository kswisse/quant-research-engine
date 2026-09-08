"""Market identity — provider-independent resolution semantics.

Phase 2.5 — Market Identity & Resolution Semantics.
"""

from quant_engine.market_identity.errors import (
    IncompatibleOutcomeError,
    IncompatibleResolutionError,
    IncompatibleResolutionStateError,
    IncompatibleSettlementError,
    MarketIdentityError,
)
from quant_engine.market_identity.mapping import MarketMapping
from quant_engine.market_identity.models import (
    EventBoundary,
    EventIdentity,
    MarketIdentity,
    OutcomeIdentity,
    ResolutionRule,
    ResolutionState,
    SettlementTerms,
    TemporalScope,
)

__all__ = [
    "EventBoundary",
    "EventIdentity",
    "IncompatibleOutcomeError",
    "IncompatibleResolutionError",
    "IncompatibleResolutionStateError",
    "IncompatibleSettlementError",
    "MarketIdentity",
    "MarketIdentityError",
    "MarketMapping",
    "OutcomeIdentity",
    "ResolutionRule",
    "ResolutionState",
    "SettlementTerms",
    "TemporalScope",
]
