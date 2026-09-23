"""Chat model selected from .env."""

import os
from contextlib import contextmanager
from functools import lru_cache
import httpx
from langchain_core.language_models.chat_models import BaseChatModel

from aqe.settings import Settings, get_settings


@contextmanager
def _ssl_verify_env(verify: bool):
    original = os.environ.get("PYTHONHTTPSVERIFY")
    if not verify:
        os.environ["PYTHONHTTPSVERIFY"] = "0"
    try:
        yield
    finally:
        if original is not None:
            os.environ["PYTHONHTTPSVERIFY"] = original
        elif "PYTHONHTTPSVERIFY" in os.environ:
            del os.environ["PYTHONHTTPSVERIFY"]


def _load_chat_class(module_name: str, class_name: str, provider: str) -> type:
    try:
        module = __import__(module_name, fromlist=[class_name])
    except ImportError as exc:
        raise RuntimeError(
            f"LLM_PROVIDER={provider} is missing its client package. Install the project with: uv pip install -e ."
        ) from exc
    return getattr(module, class_name)


def _openai_clients(settings: Settings) -> tuple[httpx.Client, httpx.AsyncClient]:
    return (
        httpx.Client(verify=settings.ssl_verify),
        httpx.AsyncClient(verify=settings.ssl_verify),
    )


@lru_cache
def get_chat_model() -> BaseChatModel:
    settings = get_settings()
    provider = settings.llm_provider
    model = settings.llm_model
    temperature = settings.llm_temperature
    if not model:
        raise ValueError("LLM_MODEL is required in .env")
    if provider == "openai":
        if settings.openai_api_key:
            ChatOpenAI = _load_chat_class("langchain_openai", "ChatOpenAI", provider)

            with _ssl_verify_env(settings.ssl_verify):
                http_client, http_async_client = _openai_clients(settings)
                return ChatOpenAI(
                    model=model,
                    api_key=settings.openai_api_key,
                    temperature=temperature,
                    http_client=http_client,
                    http_async_client=http_async_client,
                )
        if settings.aia_gateway_client_id and settings.aia_gateway_client_secret and settings.aia_gateway_base_url:
            ChatOpenAI = _load_chat_class("langchain_openai", "ChatOpenAI", provider)

            with _ssl_verify_env(settings.ssl_verify):
                try:
                    from engineer_agent.utils.auth import build_http_clients
                except ImportError:
                    from aqe.auth import build_http_clients

                http_client, http_async_client = build_http_clients(
                    settings.aia_gateway_client_id,
                    settings.aia_gateway_client_secret,
                    verify=settings.ssl_verify,
                )
                return ChatOpenAI(
                    model=model,
                    base_url=settings.aia_gateway_base_url,
                    temperature=temperature,
                    request_timeout=120,
                    http_client=http_client,
                    http_async_client=http_async_client,
                )
        if settings.reallm_base_url and settings.reallm_api_key:
            ChatOpenAI = _load_chat_class("langchain_openai", "ChatOpenAI", provider)

            with _ssl_verify_env(settings.ssl_verify):
                http_client, http_async_client = _openai_clients(settings)
                return ChatOpenAI(
                    model=model,
                    base_url=settings.reallm_base_url,
                    api_key=settings.reallm_api_key,
                    temperature=temperature,
                    http_client=http_client,
                    http_async_client=http_async_client,
                )
        raise ValueError("No valid OpenAI configuration found in .env")
    if provider == "anthropic":
        ChatAnthropic = _load_chat_class("langchain_anthropic", "ChatAnthropic", provider)
        if settings.anthropic_api_key:
            return ChatAnthropic(
                model=model,
                api_key=settings.anthropic_api_key,
                temperature=temperature,
            )
    if provider == "ollama":
        ChatOllama = _load_chat_class("langchain_ollama", "ChatOllama", provider)
        if settings.ollama_base_url:
            return ChatOllama(
                model=model,
                base_url=settings.ollama_base_url,
                temperature=temperature,
            )
    raise RuntimeError("Failed to initialize LLM client")
