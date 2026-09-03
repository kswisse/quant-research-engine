"""Typed configuration using Pydantic Settings.

All quantitative code reads configuration through this module.
No secrets or API credentials are stored here.
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and .env file."""

    model_config = {"env_prefix": "QUANT_", "env_file": ".env", "env_file_encoding": "utf-8"}

    environment: str = "development"
    log_level: str = "INFO"
    default_seed: int = 42
    data_dir: Path = Path("./data")

    def ensure_data_dir(self) -> Path:
        """Create data directory if it doesn't exist and return its path."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir


settings = Settings()
