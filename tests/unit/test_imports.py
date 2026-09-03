"""Tests for package imports and basic functionality."""

from pathlib import Path

import quant_engine
from quant_engine.core.config import settings


def test_package_version():
    assert quant_engine.__version__ == "0.1.0"


def test_settings_instantiation():
    assert settings.environment in ("development", "testing", "production")
    assert settings.default_seed == 42
    assert settings.log_level == "INFO"


def test_settings_data_dir():
    path = settings.data_dir
    assert path is not None
    assert isinstance(path, Path)
