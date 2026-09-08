"""Market identity error types."""

from quant_engine.core.errors import QuantEngineError


class MarketIdentityError(QuantEngineError):
    """Base exception for all market identity errors."""


class IncompatibleOutcomeError(MarketIdentityError):
    """Raised when outcome identities are incompatible for a mapping."""


class IncompatibleResolutionError(MarketIdentityError):
    """Raised when resolution rules are incompatible for a mapping."""


class IncompatibleResolutionStateError(MarketIdentityError):
    """Raised when resolution states are incompatible for a mapping."""


class IncompatibleSettlementError(MarketIdentityError):
    """Raised when settlement terms are incompatible for a mapping."""
