"""Dataset persistence, replay, and integrity verification."""

from quant_engine.datasets.errors import (
    CorruptedDataError,
    DatasetError,
    IntegrityError,
    ManifestError,
)
from quant_engine.datasets.io import load_dataset, replay_dataset, save_dataset
from quant_engine.datasets.models import DatasetManifest, SnapshotMetadata
from quant_engine.datasets.replay import DatasetReplay
from quant_engine.datasets.storage import ParquetStorage, StorageBackend

__all__ = [
    "CorruptedDataError",
    "DatasetError",
    "DatasetManifest",
    "DatasetReplay",
    "IntegrityError",
    "ManifestError",
    "ParquetStorage",
    "SnapshotMetadata",
    "StorageBackend",
    "load_dataset",
    "replay_dataset",
    "save_dataset",
]
