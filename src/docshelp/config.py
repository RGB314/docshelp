"""Central place for settings, model and embedding setup.

All configuration comes from environment variables (optionally loaded from a git-ignored `.env`
file), so no keys or machine-specific values ever live in the code. The rest of the app asks this
module for "the chat model" or "the embeddings" and never names a vendor.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from docshelp.console import ensure_utf8
from docshelp.providers import LLMProvider, ProviderConfigError, registry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
CACHE_DIR = PROJECT_ROOT / ".cache"

load_dotenv(PROJECT_ROOT / ".env")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
ensure_utf8()


@dataclass(frozen=True)
class Settings:
    """App settings. Build with `Settings.from_env()`; immutable once created."""

    provider: str = "anthropic"
    model: str | None = None  # None = the provider's default model
    fast_model: str | None = None  # None = the provider's smaller model
    max_tokens: int = 4096
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env

        def read(var: str) -> str | None:
            return (env.get(var) or "").strip() or None

        max_tokens = read("DOCSHELP_MAX_TOKENS")
        if max_tokens is not None and not max_tokens.isdigit():
            raise ProviderConfigError(f"DOCSHELP_MAX_TOKENS must be a whole number, got {max_tokens!r}")

        defaults = cls()
        return cls(
            provider=(read("DOCSHELP_PROVIDER") or defaults.provider).lower(),
            model=read("DOCSHELP_MODEL"),
            fast_model=read("DOCSHELP_FAST_MODEL"),
            max_tokens=int(max_tokens) if max_tokens else defaults.max_tokens,
            embedding_model=read("DOCSHELP_EMBEDDING_MODEL") or defaults.embedding_model,
        )


def get_provider(settings: Settings | None = None, env: Mapping[str, str] | None = None) -> LLMProvider:
    settings = settings or Settings.from_env(env)
    return registry.create(settings.provider, env)


def get_llm(
    model: str | None = None,
    *,
    fast: bool = False,
    settings: Settings | None = None,
    env: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Return a chat model from the configured provider (DOCSHELP_PROVIDER).

    Whatever the vendor, the result has the same LangChain interface: `invoke`, `stream`, `batch`,
    `bind_tools`, `with_structured_output`.
    """
    settings = settings or Settings.from_env(env)
    chosen = model or (settings.fast_model if fast else settings.model)
    kwargs.setdefault("max_tokens", settings.max_tokens)
    return get_provider(settings, env).chat_model(chosen, fast=fast, **kwargs)


def structured_llm(llm: BaseChatModel, schema: type, *, settings: Settings | None = None) -> Any:
    """`llm.with_structured_output(schema)` using the method the configured provider supports best."""
    return get_provider(settings).structured(llm, schema)


def describe(settings: Settings | None = None, env: Mapping[str, str] | None = None) -> str:
    """Human-readable "Provider (model)" for the configured chat model."""
    settings = settings or Settings.from_env(env)
    return get_provider(settings, env).describe(settings.model)


def ensure_llm_ready(settings: Settings | None = None, env: Mapping[str, str] | None = None) -> None:
    """Exit with setup instructions if the provider isn't installed, has no API key, or isn't running."""
    settings = settings or Settings.from_env(env)
    provider = get_provider(settings, env)
    try:
        provider.check()
        provider.check_runtime(settings.model)
    except ProviderConfigError as err:
        raise SystemExit(f"{err}\n(Choose a provider with DOCSHELP_PROVIDER in .env; see README.)") from None


class FastEmbedEmbeddings(Embeddings):
    """Local, CPU-only embeddings via `fastembed` (ONNX). No API key needed, whatever the chat provider.

    Implementing LangChain's `Embeddings` interface takes just two methods, and any vector store
    in the ecosystem can then use it.
    """

    def __init__(self, model_name: str | None = None):
        from fastembed import TextEmbedding

        self.model_name = model_name or Settings.from_env().embedding_model
        # downloads ~70 MB on first use, then loads from the local cache
        self._model = TextEmbedding(self.model_name, cache_dir=str(CACHE_DIR / "fastembed"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


def get_embeddings() -> Embeddings:
    return FastEmbedEmbeddings()


def langsmith_enabled() -> bool:
    """LangChain/LangGraph send traces to LangSmith automatically when these env vars are set."""
    tracing = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    return tracing and bool(os.getenv("LANGSMITH_API_KEY"))
