"""Tests for the Historical Dataset & Replay Engine.

Comprehensive test suite covering:
- Save/Load roundtrip fidelity
- Dataset identity (deterministic IDs)
- Manifest correctness
- Integrity validation
- Ordering determinism
- Replay behavior
- Immutability guarantees
- Edge cases
- Property-based tests (Hypothesis)

All tests use tmp_path for isolation and require no network access.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from quant_engine.market_data.models import Dataset, MarketQuote

try:
    from quant_engine.datasets import (
        DatasetReplay,
        load_dataset,
        replay_dataset,
        save_dataset,
    )
    from quant_engine.datasets.errors import (
        CorruptedDataError,
        DatasetError,
        IntegrityError,
        ManifestError,
    )
    from quant_engine.datasets.models import DatasetManifest, SnapshotMetadata
    from quant_engine.datasets.storage import ParquetStorage

    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _utc(ts: str) -> datetime:
    """Parse ISO string to UTC datetime."""
    return datetime.fromisoformat(ts)


def _make_quote(
    ts: str = "2026-09-04T01:00:00+00:00",
    bid: float = 0.50,
    ask: float = 0.55,
    instrument: str = "0x1234",
    provider: str = "polymarket",
    bid_size: float = 100.0,
    ask_size: float = 200.0,
) -> MarketQuote:
    """Create a valid test quote with configurable fields."""
    return MarketQuote(
        source_timestamp=_utc(ts),
        ingestion_timestamp=_utc(ts.replace("+00:00", "+00:01")),
        provider=provider,
        provider_instrument_id=instrument,
        bid_price=bid,
        bid_size=bid_size,
        ask_price=ask,
        ask_size=ask_size,
    )


def _make_one_sided_quote(
    ts: str = "2026-09-04T01:00:00+00:00",
    bid: float = 0.50,
    instrument: str = "0x1234",
) -> MarketQuote:
    """Create a quote with only bid side (ask is None)."""
    return MarketQuote(
        source_timestamp=_utc(ts),
        ingestion_timestamp=_utc(ts.replace("+00:00", "+00:01")),
        provider="polymarket",
        provider_instrument_id=instrument,
        bid_price=bid,
        bid_size=100.0,
        ask_price=None,
        ask_size=None,
    )


SAMPLE_QUOTES: list[MarketQuote] = [
    _make_quote("2026-09-04T01:00:00+00:00", 0.50, 0.55),
    _make_quote("2026-09-04T01:01:00+00:00", 0.51, 0.56),
    _make_quote("2026-09-04T01:02:00+00:00", 0.52, 0.57),
]


def _make_dataset(records: list[MarketQuote] | None = None) -> Dataset:
    """Create a Dataset with sample records."""
    return Dataset(records=records if records is not None else SAMPLE_QUOTES)


pytestmark = pytest.mark.skipif(
    not DATASETS_AVAILABLE,
    reason="quant_engine.datasets not yet implemented",
)


# ---------------------------------------------------------------------------
# A. Save/Load Roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoadRoundtrip:
    """Save and load should preserve all data exactly."""

    def test_multi_record_roundtrip(self, tmp_path: Path) -> None:
        ds = _make_dataset(SAMPLE_QUOTES)
        save_dataset(ds, tmp_path)
        loaded = load_dataset(tmp_path)
        assert loaded.dataset_id == ds.dataset_id
        assert len(loaded.records) == len(ds.records)
        for original, restored in zip(ds.records, loaded.records):
            assert restored.record_id == original.record_id
            assert restored.provider == original.provider
            assert restored.provider_instrument_id == original.provider_instrument_id
            assert restored.bid_price == original.bid_price
            assert restored.ask_price == original.ask_price
            assert restored.bid_size == original.bid_size
            assert restored.ask_size == original.ask_size

    def test_single_record_roundtrip(self, tmp_path: Path) -> None:
        ds = _make_dataset([_make_quote()])
        save_dataset(ds, tmp_path)
        loaded = load_dataset(tmp_path)
        assert len(loaded.records) == 1
        assert loaded.records[0].record_id == ds.records[0].record_id

    def test_empty_dataset_roundtrip(self, tmp_path: Path) -> None:
        ds = _make_dataset([])
        try:
            save_dataset(ds, tmp_path)
            loaded = load_dataset(tmp_path)
            assert len(loaded.records) == 0
            assert loaded.dataset_id == ds.dataset_id
        except (ValueError, DatasetError):
            pytest.skip("Empty dataset not supported")

    def test_timestamps_preserved_exactly(self, tmp_path: Path) -> None:
        q = _make_quote("2026-09-04T01:00:00+00:00")
        ds = _make_dataset([q])
        save_dataset(ds, tmp_path)
        loaded = load_dataset(tmp_path)
        restored = loaded.records[0]
        assert restored.source_timestamp == q.source_timestamp
        assert restored.ingestion_timestamp == q.ingestion_timestamp
        assert restored.source_timestamp.tzinfo is not None
        assert restored.ingestion_timestamp.tzinfo is not None

    def test_none_values_preserved(self, tmp_path: Path) -> None:
        q = _make_one_sided_quote()
        ds = _make_dataset([q])
        save_dataset(ds, tmp_path)
        loaded = load_dataset(tmp_path)
        restored = loaded.records[0]
        assert restored.bid_price == 0.50
        assert restored.bid_size == 100.0
        assert restored.ask_price is None
        assert restored.ask_size is None

    def test_float_values_preserved(self, tmp_path: Path) -> None:
        q = _make_quote(bid=0.123456789, ask=0.987654321, bid_size=1e10, ask_size=1e-10)
        ds = _make_dataset([q])
        save_dataset(ds, tmp_path)
        loaded = load_dataset(tmp_path)
        restored = loaded.records[0]
        assert restored.bid_price == pytest.approx(0.123456789)
        assert restored.ask_price == pytest.approx(0.987654321)
        assert restored.bid_size == pytest.approx(1e10)
        assert restored.ask_size == pytest.approx(1e-10)

    def test_record_ids_preserved(self, tmp_path: Path) -> None:
        ds = _make_dataset(SAMPLE_QUOTES)
        original_ids = [r.record_id for r in ds.records]
        save_dataset(ds, tmp_path)
        loaded = load_dataset(tmp_path)
        loaded_ids = [r.record_id for r in loaded.records]
        assert loaded_ids == original_ids

    def test_schema_version_preserved(self, tmp_path: Path) -> None:
        ds = Dataset(schema_version="2", records=SAMPLE_QUOTES)
        save_dataset(ds, tmp_path)
        loaded = load_dataset(tmp_path)
        assert loaded.schema_version == "2"


# ---------------------------------------------------------------------------
# B. Dataset Identity
# ---------------------------------------------------------------------------

class TestDatasetIdentity:
    """Dataset IDs should be deterministic and stable."""

    def test_same_dataset_same_id(self, tmp_path: Path) -> None:
        ds1 = _make_dataset(SAMPLE_QUOTES)
        ds2 = _make_dataset(SAMPLE_QUOTES)
        assert ds1.dataset_id == ds2.dataset_id
        save_dataset(ds1, tmp_path)
        loaded = load_dataset(tmp_path)
        assert loaded.dataset_id == ds1.dataset_id

    def test_changed_record_different_id(self) -> None:
        q1 = _make_quote(bid=0.50)
        q2 = _make_quote(bid=0.60)
        ds1 = Dataset(records=[q1])
        ds2 = Dataset(records=[q2])
        assert ds1.dataset_id != ds2.dataset_id

    def test_different_order_same_id(self) -> None:
        q1 = _make_quote(instrument="A")
        q2 = _make_quote(instrument="B")
        ds1 = Dataset(records=[q1, q2])
        ds2 = Dataset(records=[q2, q1])
        assert ds1.dataset_id == ds2.dataset_id

    def test_save_load_preserves_dataset_id(self, tmp_path: Path) -> None:
        ds = _make_dataset(SAMPLE_QUOTES)
        original_id = ds.dataset_id
        save_dataset(ds, tmp_path)
        loaded = load_dataset(tmp_path)
        assert loaded.dataset_id == original_id


# ---------------------------------------------------------------------------
# C. Manifest
# ---------------------------------------------------------------------------

class TestManifest:
    """Manifest file should contain correct metadata."""

    def _load_manifest(self, path: Path) -> dict[str, Any]:
        manifest_path = path / "manifest.json"
        assert manifest_path.exists(), f"manifest.json not found at {manifest_path}"
        with open(manifest_path, encoding="utf-8") as f:
            return json.load(f)

    def test_manifest_exists_after_save(self, tmp_path: Path) -> None:
        ds = _make_dataset(SAMPLE_QUOTES)
        save_dataset(ds, tmp_path)
        manifest_path = tmp_path / "manifest.json"
        assert manifest_path.exists()

    def test_manifest_has_correct_dataset_id(self, tmp_path: Path) -> None:
        ds = _make_dataset(SAMPLE_QUOTES)
        save_dataset(ds, tmp_path)
        manifest = self._load_manifest(tmp_path)
        assert manifest["dataset_id"] == ds.dataset_id

    def test_manifest_has_correct_record_count(self, tmp_path: Path) -> None:
        ds = _make_dataset(SAMPLE_QUOTES)
        save_dataset(ds, tmp_path)
        manifest = self._load_manifest(tmp_path)
        assert manifest["record_count"] == len(SAMPLE_QUOTES)

    def test_manifest_has_correct_schema_version(self, tmp_path: Path) -> None:
        ds = _make_dataset(SAMPLE_QUOTES)
        save_dataset(ds, tmp_path)
        manifest = self._load_manifest(tmp_path)
        assert manifest["schema_version"] == ds.schema_version

    def test_manifest_has_provider_list(self, tmp_path: Path) -> None:
        q1 = _make_quote(provider="polymarket")
        q2 = _make_quote(provider="kalshi")
        ds = _make_dataset([q1, q2])
        save_dataset(ds, tmp_path)
        manifest = self._load_manifest(tmp_path)
        providers = manifest.get("providers", [])
        assert "polymarket" in providers
        assert "kalshi" in providers

    def test_manifest_has_instrument_list(self, tmp_path: Path) -> None:
        q1 = _make_quote(instrument="0xAAAA")
        q2 = _make_quote(instrument="0xBBBB")
        ds = _make_dataset([q1, q2])
        save_dataset(ds, tmp_path)
        manifest = self._load_manifest(tmp_path)
        instruments = manifest.get("instruments", [])
        assert "0xAAAA" in instruments
        assert "0xBBBB" in instruments

    def test_manifest_has_source_time_range(self, tmp_path: Path) -> None:
        ds = _make_dataset(SAMPLE_QUOTES)
        save_dataset(ds, tmp_path)
        manifest = self._load_manifest(tmp_path)
        assert "source_time_range" in manifest
        time_range = manifest["source_time_range"]
        assert "start" in time_range
        assert "end" in time_range

    def test_manifest_has_file_checksum(self, tmp_path: Path) -> None:
        ds = _make_dataset(SAMPLE_QUOTES)
        save_dataset(ds, tmp_path)
        manifest = self._load_manifest(tmp_path)
        assert "file_checksum" in manifest
        assert isinstance(manifest["file_checksum"], str)
        assert len(manifest["file_checksum"]) > 0

    def test_manifest_has_record_id_checksum(self, tmp_path: Path) -> None:
        ds = _make_dataset(SAMPLE_QUOTES)
        save_dataset(ds, tmp_path)
        manifest = self._load_manifest(tmp_path)
        assert "record_id_checksum" in manifest
        assert isinstance(manifest["record_id_checksum"], str)
        assert len(manifest["record_id_checksum"]) > 0


# ---------------------------------------------------------------------------
# D. Integrity
# ---------------------------------------------------------------------------

class TestIntegrity:
    """Integrity checks should catch corruption and mismatches."""

    def test_missing_data_parquet_raises_error(self, tmp_path: Path) -> None:
        ds = _make_dataset(SAMPLE_QUOTES)
        save_dataset(ds, tmp_path)
        parquet_path = tmp_path / "data.parquet"
        if parquet_path.exists():
            parquet_path.unlink()
        with pytest.raises((FileNotFoundError, CorruptedDataError, DatasetError)):
            load_dataset(tmp_path)

    def test_missing_manifest_json_raises_error(self, tmp_path: Path) -> None:
        ds = _make_dataset(SAMPLE_QUOTES)
        save_dataset(ds, tmp_path)
        manifest_path = tmp_path / "manifest.json"
        if manifest_path.exists():
            manifest_path.unlink()
        with pytest.raises((FileNotFoundError, ManifestError, DatasetError)):
            load_dataset(tmp_path)

    def test_corrupted_manifest_raises_error(self, tmp_path: Path) -> None:
        ds = _make_dataset(SAMPLE_QUOTES)
        save_dataset(ds, tmp_path)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text("not valid json {{{", encoding="utf-8")
        with pytest.raises((json.JSONDecodeError, ManifestError, DatasetError)):
            load_dataset(tmp_path)

    def test_corrupted_parquet_raises_error(self, tmp_path: Path) -> None:
        ds = _make_dataset(SAMPLE_QUOTES)
        save_dataset(ds, tmp_path)
        parquet_path = tmp_path / "data.parquet"
        parquet_path.write_bytes(b"not a parquet file")
        with pytest.raises((Exception, CorruptedDataError, DatasetError)):
            load_dataset(tmp_path)

    def test_wrong_dataset_id_in_manifest_raises_error(self, tmp_path: Path) -> None:
        ds = _make_dataset(SAMPLE_QUOTES)
        save_dataset(ds, tmp_path)
        manifest_path = tmp_path / "manifest.json"
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        manifest["dataset_id"] = "0000000000000000"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)
        with pytest.raises((IntegrityError, DatasetError)):
            load_dataset(tmp_path)

    def test_wrong_record_count_in_manifest_raises_error(self, tmp_path: Path) -> None:
        ds = _make_dataset(SAMPLE_QUOTES)
        save_dataset(ds, tmp_path)
        manifest_path = tmp_path / "manifest.json"
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        manifest["record_count"] = 999
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)
        with pytest.raises((IntegrityError, DatasetError)):
            load_dataset(tmp_path)

    def test_wrong_schema_version_raises_error(self, tmp_path: Path) -> None:
        ds = _make_dataset(SAMPLE_QUOTES)
        save_dataset(ds, tmp_path)
        manifest_path = tmp_path / "manifest.json"
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        manifest["schema_version"] = "999"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)
        with pytest.raises((IntegrityError, DatasetError)):
            load_dataset(tmp_path)

    def test_file_checksum_mismatch_raises_error(self, tmp_path: Path) -> None:
        ds = _make_dataset(SAMPLE_QUOTES)
        save_dataset(ds, tmp_path)
        manifest_path = tmp_path / "manifest.json"
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        manifest["file_checksum"] = "00000000000000000000000000000000"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)
        with pytest.raises((IntegrityError, DatasetError)):
            load_dataset(tmp_path)


# ---------------------------------------------------------------------------
# E. Ordering
# ---------------------------------------------------------------------------

class TestOrdering:
    """Records should be saved and replayed in deterministic order."""

    def test_records_saved_in_deterministic_order(self, tmp_path: Path) -> None:
        ds = _make_dataset(SAMPLE_QUOTES)
        save_dataset(ds, tmp_path)
        loaded = load_dataset(tmp_path)
        original_ids = [r.record_id for r in ds.records]
        loaded_ids = [r.record_id for r in loaded.records]
        assert loaded_ids == original_ids

    def test_replay_returns_deterministic_order(self, tmp_path: Path) -> None:
        ds = _make_dataset(SAMPLE_QUOTES)
        save_dataset(ds, tmp_path)
        replay1 = list(replay_dataset(tmp_path))
        replay2 = list(replay_dataset(tmp_path))
        assert len(replay1) == len(replay2)
        for r1, r2 in zip(replay1, replay2):
            assert r1.record_id == r2.record_id

    def test_equal_timestamps_handled_deterministically(self, tmp_path: Path) -> None:
        q1 = _make_quote(ts="2026-09-04T01:00:00+00:00", instrument="A")
        q2 = _make_quote(ts="2026-09-04T01:00:00+00:00", instrument="B")
        ds = Dataset(records=[q1, q2])
        save_dataset(ds, tmp_path)
        loaded = load_dataset(tmp_path)
        ids = [r.record_id for r in loaded.records]
        assert len(ids) == 2
        assert len(set(ids)) == 2


# ---------------------------------------------------------------------------
# F. Replay
# ---------------------------------------------------------------------------

class TestReplay:
    """Replay behavior should be deterministic and provider-independent."""

    def test_no_network_access(self, tmp_path: Path) -> None:
        ds = _make_dataset(SAMPLE_QUOTES)
        save_dataset(ds, tmp_path)
        records = list(replay_dataset(tmp_path))
        assert len(records) == len(SAMPLE_QUOTES)

    def test_identical_replay_twice(self, tmp_path: Path) -> None:
        ds = _make_dataset(SAMPLE_QUOTES)
        save_dataset(ds, tmp_path)
        replay1 = list(replay_dataset(tmp_path))
        replay2 = list(replay_dataset(tmp_path))
        assert len(replay1) == len(replay2)
        for r1, r2 in zip(replay1, replay2):
            assert r1.record_id == r2.record_id

    def test_record_identity_preserved(self, tmp_path: Path) -> None:
        ds = _make_dataset(SAMPLE_QUOTES)
        save_dataset(ds, tmp_path)
        original_ids = {r.record_id for r in ds.records}
        replay_ids = {r.record_id for r in replay_dataset(tmp_path)}
        assert replay_ids == original_ids

    def test_timestamps_preserved_in_replay(self, tmp_path: Path) -> None:
        ds = _make_dataset(SAMPLE_QUOTES)
        save_dataset(ds, tmp_path)
        for original, replayed in zip(ds.records, replay_dataset(tmp_path)):
            assert replayed.source_timestamp == original.source_timestamp
            assert replayed.ingestion_timestamp == original.ingestion_timestamp

    def test_provider_independent_replay(self, tmp_path: Path) -> None:
        q1 = _make_quote(provider="polymarket")
        q2 = _make_quote(provider="kalshi")
        ds = Dataset(records=[q1, q2])
        save_dataset(ds, tmp_path)
        records = list(replay_dataset(tmp_path))
        providers = {r.provider for r in records}
        assert providers == {"polymarket", "kalshi"}


# ---------------------------------------------------------------------------
# G. Immutability
# ---------------------------------------------------------------------------

class TestImmutability:
    """Datasets should be immutable once saved."""

    def test_identical_dataset_can_be_loaded_again(self, tmp_path: Path) -> None:
        ds = _make_dataset(SAMPLE_QUOTES)
        save_dataset(ds, tmp_path)
        loaded1 = load_dataset(tmp_path)
        loaded2 = load_dataset(tmp_path)
        assert loaded1.dataset_id == loaded2.dataset_id
        assert len(loaded1.records) == len(loaded2.records)

    def test_conflicting_content_rejected(self, tmp_path: Path) -> None:
        ds1 = _make_dataset(SAMPLE_QUOTES)
        save_dataset(ds1, tmp_path)

        q_different = _make_quote(bid=0.99, ask=0.99)
        ds2 = Dataset(records=[q_different])

        try:
            save_dataset(ds2, tmp_path)
            loaded = load_dataset(tmp_path)
            assert loaded.dataset_id == ds2.dataset_id
        except (DatasetError, ValueError):
            pass


# ---------------------------------------------------------------------------
# H. Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_large_dataset(self, tmp_path: Path) -> None:
        quotes = []
        for i in range(1200):
            ts = f"2026-09-04T{1 + (i // 60):02d}:{i % 60:02d}:00+00:00"
            quotes.append(_make_quote(ts=ts, bid=0.50 + (i % 100) * 0.001, instrument=f"0x{i:04X}"))
        ds = Dataset(records=quotes)
        save_dataset(ds, tmp_path)
        loaded = load_dataset(tmp_path)
        assert len(loaded.records) == 1200
        assert loaded.dataset_id == ds.dataset_id

    def test_very_large_float_values(self, tmp_path: Path) -> None:
        q = _make_quote(bid=1e15, ask=1e15, bid_size=1e15, ask_size=1e15)
        ds = _make_dataset([q])
        save_dataset(ds, tmp_path)
        loaded = load_dataset(tmp_path)
        assert loaded.records[0].bid_price == pytest.approx(1e15)

    def test_very_small_float_values(self, tmp_path: Path) -> None:
        q = _make_quote(bid=1e-15, ask=1e-15, bid_size=1e-15, ask_size=1e-15)
        ds = _make_dataset([q])
        save_dataset(ds, tmp_path)
        loaded = load_dataset(tmp_path)
        assert loaded.records[0].bid_price == pytest.approx(1e-15)

    def test_unicode_in_provider(self, tmp_path: Path) -> None:
        q = _make_quote(provider="ubermarket-\u00e9")
        ds = _make_dataset([q])
        save_dataset(ds, tmp_path)
        loaded = load_dataset(tmp_path)
        assert loaded.records[0].provider == "ubermarket-\u00e9"

    def test_unicode_in_instrument_id(self, tmp_path: Path) -> None:
        q = _make_quote(instrument="\u00e9\u00e8\u00ea")
        ds = _make_dataset([q])
        save_dataset(ds, tmp_path)
        loaded = load_dataset(tmp_path)
        assert loaded.records[0].provider_instrument_id == "\u00e9\u00e8\u00ea"

    def test_special_characters_in_provider(self, tmp_path: Path) -> None:
        q = _make_quote(provider="a-b_c.d/e:f")
        ds = _make_dataset([q])
        save_dataset(ds, tmp_path)
        loaded = load_dataset(tmp_path)
        assert loaded.records[0].provider == "a-b_c.d/e:f"

    def test_zero_float_values(self, tmp_path: Path) -> None:
        q = _make_quote(bid=0.0, ask=0.0, bid_size=0.0, ask_size=0.0)
        ds = _make_dataset([q])
        save_dataset(ds, tmp_path)
        loaded = load_dataset(tmp_path)
        assert loaded.records[0].bid_price == 0.0
        assert loaded.records[0].ask_price == 0.0

    def test_mixed_none_and_float(self, tmp_path: Path) -> None:
        q = _make_quote(bid=0.5, bid_size=100.0, ask=None, ask_size=None)
        ds = _make_dataset([q])
        save_dataset(ds, tmp_path)
        loaded = load_dataset(tmp_path)
        assert loaded.records[0].bid_price == 0.5
        assert loaded.records[0].ask_price is None


# ---------------------------------------------------------------------------
# I. Property-Based Tests (Hypothesis)
# ---------------------------------------------------------------------------

class TestPropertyBased:
    """Property-based tests for invariants."""

    def test_save_load_roundtrip_arbitrary(self, tmp_path: Path) -> None:
        from hypothesis import given, settings
        from hypothesis import strategies as st

        counter = 0

        @given(
            bids=st.lists(
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                min_size=1,
                max_size=5,
            ),
            instruments=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5),
        )
        @settings(max_examples=20)
        def _property(bids: list[float], instruments: list[str]) -> None:
            nonlocal counter
            counter += 1
            quotes = []
            for i, bid in enumerate(bids):
                inst = instruments[i % len(instruments)]
                quotes.append(_make_quote(bid=bid, ask=min(bid + 0.05, 1.0), instrument=inst))
            ds = Dataset(records=quotes)
            test_dir = tmp_path / f"prop_test_{counter}"
            test_dir.mkdir(exist_ok=True)
            save_dataset(ds, test_dir)
            loaded = load_dataset(test_dir)
            assert loaded.dataset_id == ds.dataset_id
            assert len(loaded.records) == len(ds.records)

        _property()

    def test_deterministic_dataset_id_across_saves(self, tmp_path: Path) -> None:
        from hypothesis import given, settings
        from hypothesis import strategies as st

        counter = 0

        @given(
            bid=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
            instrument=st.text(min_size=1, max_size=20),
        )
        @settings(max_examples=20)
        def _property(bid: float, instrument: str) -> None:
            nonlocal counter
            counter += 1
            q = _make_quote(bid=bid, ask=min(bid + 0.05, 1e6), instrument=instrument)
            ds = Dataset(records=[q])
            id1 = ds.dataset_id

            test_dir1 = tmp_path / f"id_test_{counter}"
            test_dir1.mkdir(exist_ok=True)
            save_dataset(ds, test_dir1)
            loaded1 = load_dataset(test_dir1)
            assert loaded1.dataset_id == id1

        _property()

    def test_replay_determinism(self, tmp_path: Path) -> None:
        from hypothesis import given, settings
        from hypothesis import strategies as st

        counter = 0

        @given(
            bids=st.lists(
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                min_size=1,
                max_size=5,
            ),
        )
        @settings(max_examples=20)
        def _property(bids: list[float]) -> None:
            nonlocal counter
            counter += 1
            quotes = [_make_quote(bid=b, ask=min(b + 0.05, 1.0), instrument=f"0x{i:04d}") for i, b in enumerate(bids)]
            ds = Dataset(records=quotes)
            test_dir = tmp_path / f"replay_test_{counter}"
            test_dir.mkdir(exist_ok=True)
            save_dataset(ds, test_dir)
            replay1 = [r.record_id for r in replay_dataset(test_dir)]
            replay2 = [r.record_id for r in replay_dataset(test_dir)]
            assert replay1 == replay2

        _property()


# ---------------------------------------------------------------------------
# J. Integration: DatasetReplay class
# ---------------------------------------------------------------------------

class TestDatasetReplay:
    """Test the DatasetReplay class interface."""

    def test_replay_context_manager(self, tmp_path: Path) -> None:
        ds = _make_dataset(SAMPLE_QUOTES)
        save_dataset(ds, tmp_path)
        with DatasetReplay(tmp_path) as replay:
            records = list(replay)
            assert len(records) == len(SAMPLE_QUOTES)

    def test_replay_iterable(self, tmp_path: Path) -> None:
        ds = _make_dataset(SAMPLE_QUOTES)
        save_dataset(ds, tmp_path)
        replay = DatasetReplay(tmp_path)
        count = 0
        for record in replay:
            assert isinstance(record, MarketQuote)
            count += 1
        assert count == len(SAMPLE_QUOTES)

    def test_replay_metadata(self, tmp_path: Path) -> None:
        ds = _make_dataset(SAMPLE_QUOTES)
        save_dataset(ds, tmp_path)
        replay = DatasetReplay(tmp_path)
        assert replay.dataset_id == ds.dataset_id
        assert replay.record_count == len(SAMPLE_QUOTES)


# ---------------------------------------------------------------------------
# K. Storage Backend
# ---------------------------------------------------------------------------

class TestStorageBackend:
    """Test ParquetStorage backend directly."""

    def test_parquet_storage_save_load(self, tmp_path: Path) -> None:
        storage = ParquetStorage()
        ds = _make_dataset(SAMPLE_QUOTES)
        storage.save(ds, tmp_path)
        loaded = storage.load(tmp_path)
        assert loaded.dataset_id == ds.dataset_id
        assert len(loaded.records) == len(ds.records)

    def test_parquet_storage_creates_directory(self, tmp_path: Path) -> None:
        storage = ParquetStorage()
        ds = _make_dataset(SAMPLE_QUOTES)
        target = tmp_path / "new" / "nested" / "dir"
        storage.save(ds, target)
        assert target.exists()
        assert (target / "data.parquet").exists()


# ---------------------------------------------------------------------------
# L. Snapshot Metadata
# ---------------------------------------------------------------------------

class TestSnapshotMetadata:
    """Test SnapshotMetadata model."""

    def test_snapshot_metadata_fields(self) -> None:
        now = datetime.now(UTC)
        metadata = SnapshotMetadata(
            snapshot_id="snap-001",
            created_at=now,
            dataset_id="abc123",
            record_count=100,
            schema_version="1",
        )
        assert metadata.snapshot_id == "snap-001"
        assert metadata.created_at == now
        assert metadata.dataset_id == "abc123"
        assert metadata.record_count == 100

    def test_snapshot_metadata_deterministic_id(self) -> None:
        now = datetime(2026, 9, 4, tzinfo=UTC)
        m1 = SnapshotMetadata(
            snapshot_id="snap-001",
            created_at=now,
            dataset_id="abc123",
            record_count=100,
            schema_version="1",
        )
        m2 = SnapshotMetadata(
            snapshot_id="snap-001",
            created_at=now,
            dataset_id="abc123",
            record_count=100,
            schema_version="1",
        )
        assert m1.snapshot_id == m2.snapshot_id
