"""Arbitrage-specific error types.

Extends the core error hierarchy with arbitrage domain errors.
"""

from quant_engine.core.errors import QuantEngineError


class ArbitrageError(QuantEngineError):
    """Base exception for all arbitrage-related errors."""


class NormalizationError(ArbitrageError):
    """Raised when arbitrage opportunity normalization fails."""

class IncompatibleOpportunityTypeError(NormalizationError):
    """Raised when opportunity types cannot be normalized."""

class InvalidLegError(NormalizationError):
    """Raised when an opportunity leg has invalid fields."""
