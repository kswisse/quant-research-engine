"""Market data: provider-independent canonical data models and validation."""

from quant_engine.market_data.errors import (
    DataIntegrityError,
    InvalidQuoteError,
    InvalidTimestampError,
    MarketDataError,
    NormalizationError,
)
from quant_engine.market_data.models import Dataset, MarketQuote
from quant_engine.market_data.normalization import (
    ExampleNormalizer,
    MarketDataNormalizer,
)
from quant_engine.market_data.validation import (
    find_duplicates,
    validate_quote,
    validate_quotes,
    validate_time_order,
)

__all__ = [
    "Dataset",
    "DataIntegrityError",
    "ExampleNormalizer",
    "InvalidQuoteError",
    "InvalidTimestampError",
    "MarketDataError",
    "MarketDataNormalizer",
    "MarketQuote",
    "NormalizationError",
    "find_duplicates",
    "validate_quote",
    "validate_quotes",
    "validate_time_order",
]
