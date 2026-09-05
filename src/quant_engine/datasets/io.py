"""Convenience functions for dataset persistence and replay."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from quant_engine.datasets.replay import DatasetReplay
from quant_engine.datasets.storage import ParquetStorage

if TYPE_CHECKING:
    from quant_engine.market_data.models import Dataset

_default_storage = ParquetStorage()


def save_dataset(dataset: Dataset, path: Path) -> Path:
    """Save a dataset directly to path.

    Creates path if needed. Writes data.parquet and manifest.json
    directly into path.

    If path already exists with the same dataset_id, verifies
    content matches (idempotent). If different content under same ID,
    raises IntegrityError.

    Args:
        dataset: The Dataset to save.
        path: Directory where data.parquet and manifest.json will be written.

    Returns:
        Path to the saved dataset directory (same as path).
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    _default_storage.save(dataset, path)
    return path


def load_dataset(dataset_path: Path) -> Dataset:
    """Load a dataset from a dataset directory.

    The directory must contain data.parquet and manifest.json.

    Args:
        dataset_path: Path to the dataset directory.

    Returns:
        The reconstructed Dataset with verified integrity.
    """
    return _default_storage.load(Path(dataset_path))


def replay_dataset(dataset: Dataset | Path) -> DatasetReplay:
    """Create a deterministic replay iterator from a Dataset or Path.

    Args:
        dataset: A Dataset object or a Path to a saved dataset directory.

    Returns:
        A DatasetReplay that yields records in deterministic order.
    """
    return DatasetReplay(dataset)
