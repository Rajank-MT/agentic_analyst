from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    app_name: str = Field(
        default="Agentic Data Analyst",
        description="Application name",
    )

    environment: str = Field(
        default="development",
        description="Application environment",
    )

    max_file_size_mb: int = Field(
        default=100,
        ge=1,
        le=1024,
        description="Maximum accepted input file size in MB",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached application settings instance."""
    return Settings()