"""Application configuration settings for ModelBench.

Reads environment variables safely with sensible defaults and type validation
via Pydantic Settings.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration settings for ModelBench application."""

    model_config = SettingsConfigDict(
        env_prefix="MODELBENCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application Metadata
    app_name: str = "ModelBench"
    app_version: str = "0.1.0"
    app_env: str = "development"
    debug: bool = False

    # Server Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Benchmark Mock Configuration
    default_mock_delay_ms: float = 120.0

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app_env.lower() in {"production", "prod"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return Settings()
