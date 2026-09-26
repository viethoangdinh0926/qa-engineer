"""Environment settings loaded from .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _blank_to_none(value: object) -> object:
    if isinstance(value, str) and not value.strip():
        return None
    return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: Literal["openai", "anthropic", "ollama"] = "ollama"
    llm_model: str = ""
    llm_temperature: float = 0
    ssl_verify: bool = True
    openai_api_key: str | None = None
    aia_gateway_client_id: str | None = None
    aia_gateway_client_secret: str | None = None
    aia_gateway_base_url: str | None = None
    aia_gateway_token_url: str | None = None
    reallm_base_url: str | None = None
    reallm_api_key: str | None = None
    anthropic_api_key: str | None = None
    ollama_base_url: str | None = "http://127.0.0.1:11434"
    target_url: str = "http://127.0.0.1:8765"
    max_retries: int = 3
    spec_max_bytes: int = Field(default=100 * 1024)
    spec_path: str | None = None
    runs_dir: Path = Path("runs")
    sut_port: int = 8765
    host: str = "127.0.0.1"
    port: int = 8000
    pi_llm_model: str | None = None

    @field_validator(
        "openai_api_key",
        "aia_gateway_client_id",
        "aia_gateway_client_secret",
        "aia_gateway_base_url",
        "aia_gateway_token_url",
        "reallm_base_url",
        "reallm_api_key",
        "anthropic_api_key",
        "ollama_base_url",
        "spec_path",
        "pi_llm_model",
        mode="before",
    )
    @classmethod
    def empty_is_missing(cls, value: object) -> object:
        return _blank_to_none(value)


@lru_cache
def get_settings() -> Settings:
    return Settings()
