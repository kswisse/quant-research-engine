"""Tests for configuration module."""

from quant_engine.core.config import Settings


def test_settings_defaults():
    s = Settings()
    assert s.environment == "development"
    assert s.log_level == "INFO"
    assert s.default_seed == 42


def test_settings_data_dir_path():
    s = Settings()
    assert s.data_dir.name == "data"
