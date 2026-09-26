"""Runtime configuration for the engine."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

LlmProvider = Literal["openai", "anthropic", "ollama"]


class EngineConfig(BaseModel):
    """Selects the planner and the drivers used for a run."""

    llm: LlmProvider = "ollama"
    browser: Literal["playwright"] = "playwright"
    max_retries: int = 1
    spec_max_bytes: int = 100 * 1024
    runs_dir: Path = Field(default_factory=lambda: Path("runs"))
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    llm_temperature: float = 0
    ssl_verify: bool = True
    aia_gateway_client_id: str | None = None
    aia_gateway_client_secret: str | None = None
    aia_gateway_base_url: str | None = None
    reallm_base_url: str | None = None
    reallm_api_key: str | None = None
    anthropic_api_key: str | None = None
    ollama_base_url: str | None = "http://127.0.0.1:11434"
    target_url: str = "http://127.0.0.1:8765"
    pi_llm_model: str | None = None

    @property
    def llm_configured(self) -> bool:
        if self.llm == "openai":
            return bool(
                self.openai_api_key
                or (
                    self.aia_gateway_client_id
                    and self.aia_gateway_client_secret
                    and self.aia_gateway_base_url
                )
                or (self.reallm_base_url and self.reallm_api_key)
            )
        if self.llm == "anthropic":
            return bool(self.anthropic_api_key)
        if self.llm == "ollama":
            return bool(self.ollama_base_url)
        return False

    @classmethod
    def from_env(cls, **overrides: object) -> "EngineConfig":
        from aqe.settings import get_settings

        settings = get_settings()
        values: dict[str, object] = {
            "llm": settings.llm_provider,
            "openai_api_key": settings.openai_api_key,
            "openai_model": settings.llm_model,
            "llm_temperature": settings.llm_temperature,
            "ssl_verify": settings.ssl_verify,
            "aia_gateway_client_id": settings.aia_gateway_client_id,
            "aia_gateway_client_secret": settings.aia_gateway_client_secret,
            "aia_gateway_base_url": settings.aia_gateway_base_url,
            "reallm_base_url": settings.reallm_base_url,
            "reallm_api_key": settings.reallm_api_key,
            "anthropic_api_key": settings.anthropic_api_key,
            "ollama_base_url": settings.ollama_base_url,
            "target_url": settings.target_url,
            "max_retries": settings.max_retries,
            "spec_max_bytes": settings.spec_max_bytes,
            "runs_dir": settings.runs_dir,
            "pi_llm_model": settings.pi_llm_model if hasattr(settings, 'pi_llm_model') else None,
        }
        values.update(overrides)
        return cls.model_validate(values)
