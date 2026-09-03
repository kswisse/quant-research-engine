"""Typed error types for the quant engine."""


class QuantEngineError(Exception):
    """Base exception for all quant engine errors."""


class ConfigurationError(QuantEngineError):
    """Raised when configuration is invalid or missing."""


class ReproducibilityError(QuantEngineError):
    """Raised when deterministic behavior cannot be guaranteed."""


class DataError(QuantEngineError):
    """Raised when data is invalid, missing, or corrupted."""


class StatisticalError(QuantEngineError):
    """Raised when a statistical computation fails or produces invalid results."""
