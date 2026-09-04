"""Market data error types."""

from quant_engine.core.errors import QuantEngineError


class MarketDataError(QuantEngineError):
    """Base exception for market data errors."""


class InvalidTimestampError(MarketDataError):
    """Raised when a timestamp is invalid (naive, missing, or malformed)."""


class InvalidQuoteError(MarketDataError):
    """Raised when a market quote fails validation."""


class DataIntegrityError(MarketDataError):
    """Raised when data integrity checks fail (duplicates, out-of-order)."""


class NormalizationError(MarketDataError):
    """Raised when normalization from raw provider data fails."""
