"""Storage backends for dataset persistence."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from quant_engine.datasets.errors import (
    CorruptedDataError,
    IntegrityError,
    ManifestError,
)
from quant_engine.datasets.models import DatasetManifest
from quant_engine.market_data.models import Dataset, MarketQuote


class StorageBackend(ABC):
    """Abstract base class for dataset storage backends."""

    @abstractmethod
    def save(self, dataset: Dataset, path: Path) -> DatasetManifest:
        """Persist a dataset to the given path."""
        ...

    @abstractmethod
    def load(self, path: Path) -> Dataset:
        """Load a dataset from the given path."""
        ...


class ParquetStorage(StorageBackend):
    """Storage backend using Polars Parquet format.

    Writes one row per MarketQuote record. Source timestamps stored as
    ISO strings to preserve timezone info. Records sorted by
    (source_timestamp, record_id) before writing.
    """

    def save(self, dataset: Dataset, path: Path) -> DatasetManifest:
        """Save dataset to path.

        If path already exists with same dataset_id, returns existing manifest.
        If path exists with different dataset_id, raises IntegrityError.
        Uses atomic write: writes to a sibling temp dir, then renames.
        """
        path = Path(path)

        if path.exists():
            manifest_path = path / "manifest.json"
            if manifest_path.exists():
                try:
                    existing = DatasetManifest.model_validate_json(
                        manifest_path.read_text(encoding="utf-8")
                    )
                except Exception as e:
                    raise ManifestError(
                        f"Cannot read existing manifest at {manifest_path}: {e}"
                    ) from e
                if existing.dataset_id == dataset.dataset_id:
                    return existing
                raise IntegrityError(
                    f"Directory {path} already exists with dataset_id "
                    f"{existing.dataset_id}, cannot save {dataset.dataset_id}"
                )

        parent = path.parent
        parent.mkdir(parents=True, exist_ok=True)

        tmpdir = Path(tempfile.mkdtemp(dir=parent, prefix=".tmp_"))
        try:
            tmp_path = tmpdir / path.name
            tmp_path.mkdir()

            self._write_parquet(dataset, tmp_path / "data.parquet")
            manifest = self._create_manifest(dataset, tmp_path / "data.parquet")
            (tmp_path / "manifest.json").write_text(
                manifest.model_dump_json(indent=2), encoding="utf-8"
            )

            if path.exists():
                shutil.rmtree(path)
            shutil.move(str(tmp_path), str(path))
        finally:
            if tmpdir.exists():
                shutil.rmtree(tmpdir, ignore_errors=True)

        return manifest

    def load(self, path: Path) -> Dataset:
        """Load dataset from path with integrity verification."""
        path = Path(path)

        if not path.exists():
            raise CorruptedDataError(f"Dataset directory does not exist: {path}")

        parquet_path = path / "data.parquet"
        manifest_path = path / "manifest.json"

        if not parquet_path.exists():
            raise CorruptedDataError(f"Missing data.parquet in {path}")
        if not manifest_path.exists():
            raise ManifestError(f"Missing manifest.json in {path}")

        try:
            manifest = DatasetManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except Exception as e:
            raise ManifestError(f"Invalid manifest at {manifest_path}: {e}") from e

        actual_checksum = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
        if actual_checksum != manifest.file_checksum:
            raise IntegrityError(
                f"File checksum mismatch: expected {manifest.file_checksum}, "
                f"got {actual_checksum}"
            )

        try:
            df = pl.read_parquet(parquet_path)
        except Exception as e:
            raise CorruptedDataError(
                f"Cannot read Parquet file at {parquet_path}: {e}"
            ) from e

        records = self._df_to_records(df)
        dataset = Dataset(schema_version=manifest.schema_version, records=records)

        if dataset.dataset_id != manifest.dataset_id:
            raise IntegrityError(
                f"Dataset ID mismatch: manifest says {manifest.dataset_id}, "
                f"computed {dataset.dataset_id}"
            )

        if len(records) != manifest.record_count:
            raise IntegrityError(
                f"Record count mismatch: manifest says {manifest.record_count}, "
                f"loaded {len(records)}"
            )

        return dataset

    def _sort_records(self, dataset: Dataset) -> list[MarketQuote]:
        """Sort records by (source_timestamp, record_id) for determinism."""
        records_with_id = [(r, r.record_id) for r in dataset.records]
        records_with_id.sort(key=lambda x: (x[0].source_timestamp.isoformat(), x[1]))
        return [r for r, _ in records_with_id]

    def _write_parquet(self, dataset: Dataset, path: Path) -> None:
        """Write dataset records to Parquet file."""
        sorted_records = self._sort_records(dataset)

        rows = []
        for r in sorted_records:
            rows.append({
                "source_timestamp": r.source_timestamp.isoformat(),
                "ingestion_timestamp": r.ingestion_timestamp.isoformat(),
                "provider": r.provider,
                "provider_instrument_id": r.provider_instrument_id,
                "bid_price": r.bid_price,
                "bid_size": r.bid_size,
                "ask_price": r.ask_price,
                "ask_size": r.ask_size,
                "schema_version": r.schema_version,
            })

        df = pl.DataFrame(rows)
        df.write_parquet(path)

    def _create_manifest(
        self, dataset: Dataset, parquet_path: Path
    ) -> DatasetManifest:
        """Create manifest from dataset and parquet file."""
        sorted_records = self._sort_records(dataset)

        if sorted_records:
            first_ts = sorted_records[0].source_timestamp.isoformat()
            last_ts = sorted_records[-1].source_timestamp.isoformat()
        else:
            first_ts = ""
            last_ts = ""

        providers = sorted({r.provider for r in dataset.records})
        instruments = sorted({r.provider_instrument_id for r in dataset.records})

        record_ids = [r.record_id for r in sorted_records]
        record_id_checksum = hashlib.sha256("".join(record_ids).encode()).hexdigest()
        file_checksum = hashlib.sha256(parquet_path.read_bytes()).hexdigest()

        return DatasetManifest(
            dataset_id=dataset.dataset_id,
            schema_version=dataset.schema_version,
            record_count=len(dataset.records),
            source_time_range={"start": first_ts, "end": last_ts},
            providers=providers,
            instruments=instruments,
            created_at=datetime.now(UTC).isoformat(),
            file_checksum=file_checksum,
            record_id_checksum=record_id_checksum,
        )

    def _df_to_records(self, df: pl.DataFrame) -> list[MarketQuote]:
        """Convert Polars DataFrame to list of MarketQuote records."""
        records = []
        for row in df.iter_rows(named=True):
            bid_price = row["bid_price"]
            bid_size = row["bid_size"]
            ask_price = row["ask_price"]
            ask_size = row["ask_size"]

            records.append(
                MarketQuote(
                    source_timestamp=datetime.fromisoformat(row["source_timestamp"]),
                    ingestion_timestamp=datetime.fromisoformat(
                        row["ingestion_timestamp"]
                    ),
                    provider=row["provider"],
                    provider_instrument_id=row["provider_instrument_id"],
                    bid_price=float(bid_price) if bid_price is not None else None,
                    bid_size=float(bid_size) if bid_size is not None else None,
                    ask_price=float(ask_price) if ask_price is not None else None,
                    ask_size=float(ask_size) if ask_size is not None else None,
                    schema_version=row["schema_version"],
                )
            )
        return records
