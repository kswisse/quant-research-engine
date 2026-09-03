"""Tests for error types."""

import pytest

from quant_engine.core.errors import (
    ConfigurationError,
    DataError,
    QuantEngineError,
    ReproducibilityError,
    StatisticalError,
)


def test_errors_are_quant_engine_error():
    assert issubclass(ConfigurationError, QuantEngineError)
    assert issubclass(ReproducibilityError, QuantEngineError)
    assert issubclass(DataError, QuantEngineError)
    assert issubclass(StatisticalError, QuantEngineError)


def test_errors_can_be_raised():
    with pytest.raises(QuantEngineError):
        raise QuantEngineError("base error")

    with pytest.raises(ConfigurationError):
        raise ConfigurationError("config error")

    with pytest.raises(StatisticalError):
        raise StatisticalError("stat error")
