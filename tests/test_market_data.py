"""Tests for market data models and validation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quant_engine.market_data import (
    Dataset,
    InvalidTimestampError,
    MarketDataError,
    MarketQuote,
)


def _utc(dt_str: str) -> datetime:
    """Parse ISO string to UTC datetime."""
    return datetime.fromisoformat(dt_str)


def _make_quote(
    provider: str = "polymarket",
    instrument: str = "0x1234",
    source_ts: str = "2026-09-04T01:00:00+00:00",
    ingest_ts: str = "2026-09-04T01:00:01+00:00",
    bid: float = 0.5,
    ask: float = 0.6,
) -> MarketQuote:
    """Create a valid test quote."""
    return MarketQuote(
        source_timestamp=_utc(source_ts),
        ingestion_timestamp=_utc(ingest_ts),
        provider=provider,
        provider_instrument_id=instrument,
        bid_price=bid,
        bid_size=100.0,
        ask_price=ask,
        ask_size=200.0,
    )


class TestMarketQuote:
    """Test MarketQuote model."""

    def test_valid_quote(self) -> None:
        q = _make_quote()
        assert q.provider == "polymarket"
        assert q.bid_price == 0.5
        assert q.ask_price == 0.6

    def test_none_sides(self) -> None:
        q = MarketQuote(
            source_timestamp=_utc("2026-09-04T01:00:00+00:00"),
            ingestion_timestamp=_utc("2026-09-04T01:00:01+00:00"),
            provider="test",
            provider_instrument_id="X",
        )
        assert q.bid_price is None
        assert q.ask_price is None

    def test_rejects_naive_source_timestamp(self) -> None:
        with pytest.raises(InvalidTimestampError):
            MarketQuote(
                source_timestamp=datetime(2026, 9, 4, 1, 0, 0),  # naive!
                ingestion_timestamp=_utc("2026-09-04T01:00:01+00:00"),
                provider="test",
                provider_instrument_id="X",
            )

    def test_rejects_naive_ingestion_timestamp(self) -> None:
        with pytest.raises(InvalidTimestampError):
            MarketQuote(
                source_timestamp=_utc("2026-09-04T01:00:00+00:00"),
                ingestion_timestamp=datetime(2026, 9, 4, 1, 0, 1),  # naive!
                provider="test",
                provider_instrument_id="X",
            )

    def test_rejects_empty_provider(self) -> None:
        with pytest.raises(Exception):
            MarketQuote(
                source_timestamp=_utc("2026-09-04T01:00:00+00:00"),
                ingestion_timestamp=_utc("2026-09-04T01:00:01+00:00"),
                provider="",
                provider_instrument_id="X",
            )

    def test_rejects_empty_instrument(self) -> None:
        with pytest.raises(Exception):
            MarketQuote(
                source_timestamp=_utc("2026-09-04T01:00:00+00:00"),
                ingestion_timestamp=_utc("2026-09-04T01:00:01+00:00"),
                provider="test",
                provider_instrument_id="",
            )

    def test_record_id_deterministic(self) -> None:
        q1 = _make_quote()
        q2 = _make_quote()
        assert q1.record_id == q2.record_id

    def test_record_id_changes_with_payload(self) -> None:
        q1 = _make_quote(bid=0.5)
        q2 = _make_quote(bid=0.6)
        assert q1.record_id != q2.record_id

    def test_record_id_changes_with_provider(self) -> None:
        q1 = _make_quote(provider="polymarket")
        q2 = _make_quote(provider="kalshi")
        assert q1.record_id != q2.record_id

    def test_serialization_roundtrip(self) -> None:
        q = _make_quote()
        d = q.to_dict()
        restored = MarketQuote.from_dict(d)
        assert restored.record_id == q.record_id
        assert restored.provider == q.provider
        assert restored.bid_price == q.bid_price

    def test_to_dict_includes_record_id(self) -> None:
        q = _make_quote()
        d = q.to_dict()
        assert "record_id" in d
        assert d["record_id"] == q.record_id

    def test_frozen(self) -> None:
        q = _make_quote()
        with pytest.raises(Exception):
            q.bid_price = 0.7  # type: ignore[misc]


class TestDataset:
    """Test Dataset model."""

    def test_valid_dataset(self) -> None:
        q1 = _make_quote(instrument="A")
        q2 = _make_quote(instrument="B")
        ds = Dataset(records=[q1, q2])
        assert len(ds.records) == 2

    def test_dataset_id_deterministic(self) -> None:
        q1 = _make_quote(instrument="A")
        q2 = _make_quote(instrument="B")
        ds1 = Dataset(records=[q1, q2])
        ds2 = Dataset(records=[q1, q2])
        assert ds1.dataset_id == ds2.dataset_id

    def test_dataset_id_changes_with_reorder(self) -> None:
        q1 = _make_quote(instrument="A")
        q2 = _make_quote(instrument="B")
        ds1 = Dataset(records=[q1, q2])
        ds2 = Dataset(records=[q2, q1])
        assert ds1.dataset_id != ds2.dataset_id

    def test_dataset_id_changes_with_record(self) -> None:
        q1 = _make_quote(instrument="A")
        q2a = _make_quote(instrument="B", bid=0.5)
        q2b = _make_quote(instrument="B", bid=0.6)
        ds1 = Dataset(records=[q1, q2a])
        ds2 = Dataset(records=[q1, q2b])
        assert ds1.dataset_id != ds2.dataset_id

    def test_serialization_roundtrip(self) -> None:
        q = _make_quote()
        ds = Dataset(records=[q])
        d = ds.to_dict()
        restored = Dataset.from_dict(d)
        assert restored.dataset_id == ds.dataset_id

    def test_frozen(self) -> None:
        q = _make_quote()
        ds = Dataset(records=[q])
        with pytest.raises(Exception):
            ds.records = []  # type: ignore[misc]


class TestErrorHierarchy:
    """Test error types."""

    def test_market_data_error_is_quant_engine_error(self) -> None:
        from quant_engine.core.errors import QuantEngineError

        assert issubclass(MarketDataError, QuantEngineError)

    def test_invalid_timestamp_is_market_data_error(self) -> None:
        assert issubclass(InvalidTimestampError, MarketDataError)


class TestValidateQuote:
    """Test single-quote validation."""

    def test_valid_quote_passes(self) -> None:
        from quant_engine.market_data.validation import validate_quote
        q = _make_quote(bid=0.5, ask=0.6)
        assert validate_quote(q) == []

    def test_rejects_negative_bid_price(self) -> None:
        from quant_engine.market_data.validation import validate_quote
        q = _make_quote()
        q2 = q.model_copy(update={"bid_price": -0.1})
        errors = validate_quote(q2)
        assert any("negative" in e for e in errors)

    def test_rejects_negative_ask_price(self) -> None:
        from quant_engine.market_data.validation import validate_quote
        q = _make_quote()
        q2 = q.model_copy(update={"ask_price": -0.1})
        errors = validate_quote(q2)
        assert any("negative" in e for e in errors)

    def test_rejects_nan_price(self) -> None:
        from quant_engine.market_data.validation import validate_quote
        q = _make_quote()
        q2 = q.model_copy(update={"bid_price": float("nan")})
        errors = validate_quote(q2)
        assert any("NaN" in e for e in errors)

    def test_rejects_inf_price(self) -> None:
        from quant_engine.market_data.validation import validate_quote
        q = _make_quote()
        q2 = q.model_copy(update={"ask_price": float("inf")})
        errors = validate_quote(q2)
        assert any("infinity" in e for e in errors)

    def test_rejects_crossed_market(self) -> None:
        from quant_engine.market_data.validation import validate_quote
        q = _make_quote(bid=0.7, ask=0.5)
        errors = validate_quote(q)
        assert any("crossed" in e for e in errors)

    def test_rejects_naive_source_timestamp(self) -> None:
        from quant_engine.market_data.validation import validate_quote
        q = _make_quote()
        q2 = q.model_copy(update={"source_timestamp": datetime(2026, 9, 4, 1, 0, 0)})
        errors = validate_quote(q2)
        assert any("naive" in e for e in errors)

    def test_valid_with_none_sides(self) -> None:
        from quant_engine.market_data.validation import validate_quote
        q = MarketQuote(
            source_timestamp=_utc("2026-09-04T01:00:00+00:00"),
            ingestion_timestamp=_utc("2026-09-04T01:00:01+00:00"),
            provider="test",
            provider_instrument_id="X",
        )
        assert validate_quote(q) == []


class TestValidateTimeOrder:
    """Test sequence-level time ordering."""

    def test_ordered_passes(self) -> None:
        from quant_engine.market_data.validation import validate_time_order
        q1 = _make_quote(source_ts="2026-09-04T01:00:00+00:00")
        q2 = _make_quote(source_ts="2026-09-04T01:00:01+00:00")
        q3 = _make_quote(source_ts="2026-09-04T01:00:02+00:00")
        assert validate_time_order([q1, q2, q3]) == []

    def test_out_of_order_detected(self) -> None:
        from quant_engine.market_data.validation import validate_time_order
        q1 = _make_quote(source_ts="2026-09-04T01:00:02+00:00")
        q2 = _make_quote(source_ts="2026-09-04T01:00:01+00:00")
        violations = validate_time_order([q1, q2])
        assert len(violations) == 1
        assert violations[0]["index"] == 1

    def test_same_timestamp_allowed(self) -> None:
        from quant_engine.market_data.validation import validate_time_order
        q1 = _make_quote(source_ts="2026-09-04T01:00:00+00:00", instrument="A")
        q2 = _make_quote(source_ts="2026-09-04T01:00:00+00:00", instrument="B")
        assert validate_time_order([q1, q2]) == []

    def test_empty_list(self) -> None:
        from quant_engine.market_data.validation import validate_time_order
        assert validate_time_order([]) == []


class TestFindDuplicates:
    """Test duplicate detection."""

    def test_no_duplicates(self) -> None:
        from quant_engine.market_data.validation import find_duplicates
        q1 = _make_quote(instrument="A")
        q2 = _make_quote(instrument="B")
        assert find_duplicates([q1, q2]) == []

    def test_duplicate_detected(self) -> None:
        from quant_engine.market_data.validation import find_duplicates
        q1 = _make_quote(instrument="A")
        q2 = _make_quote(instrument="A")
        dups = find_duplicates([q1, q2])
        assert len(dups) == 1
        assert dups[0]["indices"] == [0, 1]

    def test_same_timestamp_different_payload_not_duplicate(self) -> None:
        from quant_engine.market_data.validation import find_duplicates
        q1 = _make_quote(source_ts="2026-09-04T01:00:00+00:00", bid=0.5)
        q2 = _make_quote(source_ts="2026-09-04T01:00:00+00:00", bid=0.6)
        assert find_duplicates([q1, q2]) == []


class TestNormalization:
    """Test normalization abstraction."""

    def test_example_normalizer_produces_valid_quote(self) -> None:
        from quant_engine.market_data.normalization import ExampleNormalizer
        from quant_engine.market_data.validation import validate_quote

        normalizer = ExampleNormalizer()
        raw = {
            "provider": "example",
            "instrument_id": "BTC-USD",
            "timestamp": "2026-09-04T01:00:00+00:00",
            "bid": 0.5,
            "bid_size": 100.0,
            "ask": 0.6,
            "ask_size": 200.0,
        }
        quote = normalizer.normalize(raw)
        assert quote.provider == "example"
        assert quote.provider_instrument_id == "BTC-USD"
        assert quote.bid_price == 0.5
        assert validate_quote(quote) == []

    def test_example_normalizer_deterministic(self) -> None:
        from quant_engine.market_data.normalization import ExampleNormalizer

        normalizer = ExampleNormalizer()
        raw = {
            "provider": "example",
            "instrument_id": "X",
            "timestamp": "2026-09-04T01:00:00+00:00",
            "bid": 0.5,
            "ask": 0.6,
        }
        q1 = normalizer.normalize(raw)
        q2 = normalizer.normalize(raw)
        assert q1.record_id == q2.record_id

    def test_example_normalizer_none_sides(self) -> None:
        from quant_engine.market_data.normalization import ExampleNormalizer

        normalizer = ExampleNormalizer()
        raw = {
            "provider": "example",
            "instrument_id": "X",
            "timestamp": "2026-09-04T01:00:00+00:00",
        }
        quote = normalizer.normalize(raw)
        assert quote.bid_price is None
        assert quote.ask_price is None

    def test_rejects_missing_provider(self) -> None:
        from quant_engine.market_data.normalization import ExampleNormalizer
        from quant_engine.market_data.errors import NormalizationError

        normalizer = ExampleNormalizer()
        raw = {"instrument_id": "X", "timestamp": "2026-09-04T01:00:00+00:00"}
        with pytest.raises(NormalizationError):
            normalizer.normalize(raw)

    def test_rejects_missing_instrument(self) -> None:
        from quant_engine.market_data.normalization import ExampleNormalizer
        from quant_engine.market_data.errors import NormalizationError

        normalizer = ExampleNormalizer()
        raw = {"provider": "example", "timestamp": "2026-09-04T01:00:00+00:00"}
        with pytest.raises(NormalizationError):
            normalizer.normalize(raw)

    def test_rejects_missing_timestamp(self) -> None:
        from quant_engine.market_data.normalization import ExampleNormalizer
        from quant_engine.market_data.errors import NormalizationError

        normalizer = ExampleNormalizer()
        raw = {"provider": "example", "instrument_id": "X"}
        with pytest.raises(NormalizationError):
            normalizer.normalize(raw)

    def test_provenance_preserved(self) -> None:
        """ingestion_timestamp is set by the normalizer, not from raw data."""
        from quant_engine.market_data.normalization import ExampleNormalizer

        normalizer = ExampleNormalizer()
        raw = {
            "provider": "example",
            "instrument_id": "X",
            "timestamp": "2026-09-04T01:00:00+00:00",
        }
        quote = normalizer.normalize(raw)
        assert quote.source_timestamp.isoformat() == "2026-09-04T01:00:00+00:00"
        assert quote.ingestion_timestamp.tzinfo is not None


class TestEdgeCases:
    """Edge case tests."""

    def test_record_id_deterministic_across_instances(self) -> None:
        """Same payload → same record_id, every time."""
        for _ in range(10):
            q = _make_quote(bid=0.5, ask=0.6)
            q2 = _make_quote(bid=0.5, ask=0.6)
            assert q.record_id == q2.record_id

    def test_dataset_empty(self) -> None:
        ds = Dataset(records=[])
        assert ds.dataset_id is not None
        assert len(ds.records) == 0

    def test_single_record_dataset(self) -> None:
        q = _make_quote()
        ds = Dataset(records=[q])
        assert len(ds.records) == 1
        assert ds.dataset_id is not None

    def test_validate_quotes_multiple_errors(self) -> None:
        from quant_engine.market_data.validation import validate_quotes
        q1 = _make_quote(bid=0.7, ask=0.5)  # crossed
        q2 = _make_quote()
        errors = validate_quotes([q1, q2])
        assert 0 in errors  # q1 has errors
        assert 1 not in errors  # q2 is valid

    def test_find_duplicates_empty(self) -> None:
        from quant_engine.market_data.validation import find_duplicates
        assert find_duplicates([]) == []

    def test_validate_time_order_single_record(self) -> None:
        from quant_engine.market_data.validation import validate_time_order
        q = _make_quote()
        assert validate_time_order([q]) == []

    def test_serialization_roundtrip_preserves_record_id(self) -> None:
        q = _make_quote()
        d = q.to_dict()
        restored = MarketQuote.from_dict(d)
        assert restored.record_id == q.record_id

    def test_dataset_serialization_roundtrip(self) -> None:
        q1 = _make_quote(instrument="A")
        q2 = _make_quote(instrument="B")
        ds = Dataset(records=[q1, q2])
        d = ds.to_dict()
        restored = Dataset.from_dict(d)
        assert restored.dataset_id == ds.dataset_id
        assert len(restored.records) == 2
