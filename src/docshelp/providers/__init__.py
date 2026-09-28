"""Pluggable chat-model providers (Anthropic, OpenAI, Google Gemini, Groq, Ollama)."""

from docshelp.providers import vendors as _vendors  # noqa: F401  (registers the built-in providers)
from docshelp.providers.base import LLMProvider, ProviderConfigError, ProviderRegistry, registry

__all__ = ["LLMProvider", "ProviderConfigError", "ProviderRegistry", "registry"]
