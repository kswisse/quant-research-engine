"""Deterministic replay engine for persisted datasets."""

from __future__ import annotations

from collections.abc import Iterator  # noqa: TC003 — used in return types
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quant_engine.market_data.models import Dataset, MarketQuote


class DatasetReplay:
    """Provider-independent deterministic replay of dataset records.

    Emits records in deterministic order (source_timestamp, record_id).
    Never contacts any provider. Never mutates records.

    Accepts either a Dataset object or a Path to load from disk.
    """

    def __init__(self, source: Dataset | Path) -> None:
        """Initialize replay from a Dataset or Path.

        Args:
            source: A Dataset object or a Path to a saved dataset directory.
        """
        if isinstance(source, Path):
            from quant_engine.datasets.io import load_dataset

            dataset = load_dataset(source)
        else:
            dataset = source

        self._dataset_id = dataset.dataset_id
        self._record_count = len(dataset.records)

        records_with_id = [(r, r.record_id) for r in dataset.records]
        records_with_id.sort(
            key=lambda x: (
                x[0].ingestion_timestamp.isoformat()
                if x[0].source_timestamp_missing
                else x[0].source_timestamp.isoformat(),
                x[1],
            )
        )
        self._records = [r for r, _ in records_with_id]

    @property
    def dataset_id(self) -> str:
        """The dataset identifier."""
        return self._dataset_id

    @property
    def record_count(self) -> int:
        """Number of records in the replay."""
        return self._record_count

    def records(self) -> Iterator[MarketQuote]:
        """Yield records in deterministic order."""
        yield from self._records

    def __iter__(self) -> Iterator[MarketQuote]:
        """Iterate over records in deterministic order."""
        yield from self._records

    def __enter__(self) -> DatasetReplay:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit."""
