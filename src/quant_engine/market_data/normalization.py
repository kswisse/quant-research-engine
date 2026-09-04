"""Provider-independent normalization interface.

Defines the MarketDataNormalizer Protocol for converting raw provider records
into canonical MarketQuote records. No provider-specific schema should leak
into the research engine.

Architecture:
    Provider Adapter (future)
        ↓
    Raw Provider Record (dict/typed)
        ↓
    MarketDataNormalizer.normalize()
        ↓
    MarketQuote (canonical, validated)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

from quant_engine.market_data.errors import NormalizationError
from quant_engine.market_data.models import MarketQuote


class MarketDataNormalizer(ABC):
    """Abstract base class for provider-specific normalizers.

    Each provider adapter implements this interface to convert
    provider-specific raw records into canonical MarketQuote records.

    The canonical layer never knows about Polymarket JSON, Kalshi JSON,
    or any other provider-specific format. It only knows MarketQuote.
    """

    @abstractmethod
    def normalize(self, raw_record: dict[str, Any]) -> MarketQuote:
        """Normalize a raw provider record into a canonical MarketQuote.

        Args:
            raw_record: Provider-specific raw data as a dictionary.

        Returns:
            Canonical MarketQuote.

        Raises:
            NormalizationError: If the raw record cannot be normalized.
        """
        ...


class ExampleNormalizer(MarketDataNormalizer):
    """Example normalizer for testing and demonstration.

    Converts a simple flat dictionary into a MarketQuote.
    This is NOT a real provider adapter — it demonstrates the interface.

    Expected raw_record format:
        {
            "provider": "example",
            "instrument_id": "BTC-USD",
            "timestamp": "2026-09-04T01:00:00+00:00",
            "bid": 0.5,
            "bid_size": 100.0,
            "ask": 0.6,
            "ask_size": 200.0
        }
    """

    def normalize(self, raw_record: dict[str, Any]) -> MarketQuote:
        """Normalize an example raw record.

        Args:
            raw_record: Flat dictionary with provider-specific fields.

        Returns:
            Canonical MarketQuote.

        Raises:
            NormalizationError: If required fields are missing or invalid.
        """
        try:
            provider = str(raw_record.get("provider", ""))
            if not provider:
                raise NormalizationError("Missing required field: provider")

            instrument_id = str(raw_record.get("instrument_id", ""))
            if not instrument_id:
                raise NormalizationError("Missing required field: instrument_id")

            timestamp_str = raw_record.get("timestamp")
            if not timestamp_str:
                raise NormalizationError("Missing required field: timestamp")

            source_ts = datetime.fromisoformat(str(timestamp_str))
            if source_ts.tzinfo is None:
                source_ts = source_ts.replace(tzinfo=UTC)

            return MarketQuote(
                source_timestamp=source_ts,
                ingestion_timestamp=datetime.now(UTC),
                provider=provider,
                provider_instrument_id=instrument_id,
                bid_price=float(raw_record["bid"]) if "bid" in raw_record else None,
                bid_size=float(raw_record["bid_size"]) if "bid_size" in raw_record else None,
                ask_price=float(raw_record["ask"]) if "ask" in raw_record else None,
                ask_size=float(raw_record["ask_size"]) if "ask_size" in raw_record else None,
            )
        except NormalizationError:
            raise
        except Exception as e:
            raise NormalizationError(f"Failed to normalize record: {e}") from e
