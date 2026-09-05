"""Dataset metadata models for persistence and provenance tracking."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 — Pydantic needs this at runtime

from pydantic import BaseModel


class DatasetManifest(BaseModel):
    """Immutable metadata about a persisted dataset.

    Stored alongside data.parquet in each dataset directory.
    Provides integrity verification, provenance, and quick summary
    without reading the full Parquet file.
    """

    model_config = {"frozen": True}

    dataset_id: str
    schema_version: str
    record_count: int
    source_time_range: dict[str, str]  # {"start": ISO, "end": ISO}
    providers: list[str]
    instruments: list[str]
    storage_format: str = "parquet"
    created_at: str  # ISO format
    file_checksum: str  # SHA-256 of data.parquet
    record_id_checksum: str  # SHA-256 of concatenated record_ids
    application_version: str = "0.1.0"


class SnapshotMetadata(BaseModel):
    """Lightweight snapshot metadata for quick identification."""

    model_config = {"frozen": True}

    snapshot_id: str
    created_at: datetime
    dataset_id: str
    record_count: int
    schema_version: str
