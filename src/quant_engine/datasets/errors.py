"""Dataset-specific error types."""

from __future__ import annotations

from quant_engine.core.errors import QuantEngineError


class DatasetError(QuantEngineError):
    """Base exception for dataset persistence/replay errors."""


class CorruptedDataError(DatasetError):
    """Raised when a dataset directory exists but is incomplete or corrupted."""


class IntegrityError(DatasetError):
    """Raised when integrity checks fail (checksum mismatch, ID mismatch)."""


class ManifestError(DatasetError):
    """Raised when the manifest is missing, invalid, or corrupted."""
