"""Concrete providers. Each one only states its facts and how to construct its chat model."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel

from docshelp.providers.base import LLMProvider, ProviderConfigError, registry


@registry.register
class AnthropicProvider(LLMProvider):
    name = "anthropic"
    display_name = "Anthropic"
    extra = "anthropic"
    module = "langchain_anthropic"
    api_key_envs = ("ANTHROPIC_API_KEY",)
    default_model = "claude-sonnet-5"
    fast_model = "claude-haiku-4-5"
    signup_url = "https://console.anthropic.com/settings/keys"

    def _create(self, model: str, max_tokens: int, **kwargs: Any) -> BaseChatModel:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, api_key=self.api_key(), max_tokens=max_tokens, **kwargs)


@registry.register
class OpenAIProvider(LLMProvider):
    name = "openai"
    display_name = "OpenAI"
    extra = "openai"
    module = "langchain_openai"
    api_key_envs = ("OPENAI_API_KEY",)
    default_model = "gpt-5.4-mini"
    fast_model = "gpt-5-mini"
    signup_url = "https://platform.openai.com/api-keys"

    def _create(self, model: str, max_tokens: int, **kwargs: Any) -> BaseChatModel:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, api_key=self.api_key(), max_tokens=max_tokens, **kwargs)


@registry.register
class GoogleProvider(LLMProvider):
    name = "google"
    display_name = "Google Gemini"
    extra = "google"
    module = "langchain_google_genai"
    api_key_envs = ("GOOGLE_API_KEY", "GEMINI_API_KEY")
    default_model = "gemini-3.8-flash"
    fast_model = "gemini-3.5-flash-lite"
    signup_url = "https://aistudio.google.com/apikey"

    def _create(self, model: str, max_tokens: int, **kwargs: Any) -> BaseChatModel:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, api_key=self.api_key(), max_tokens=max_tokens, **kwargs)


@registry.register
class GroqProvider(LLMProvider):
    name = "groq"
    display_name = "Groq"
    extra = "groq"
    module = "langchain_groq"
    api_key_envs = ("GROQ_API_KEY",)
    default_model = "openai/gpt-oss-120b"
    fast_model = "openai/gpt-oss-20b"
    signup_url = "https://console.groq.com/keys"
    # Groq supports native JSON-schema output on only some models; tool calling works on all of them.
    structured_output_method = "function_calling"

    def _create(self, model: str, max_tokens: int, **kwargs: Any) -> BaseChatModel:
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, api_key=self.api_key(), max_tokens=max_tokens, **kwargs)


@registry.register
class OllamaProvider(LLMProvider):
    """Runs models locally with Ollama: free, private, no API key. Needs the Ollama app running."""

    name = "ollama"
    display_name = "Ollama"
    extra = "ollama"
    module = "langchain_ollama"
    api_key_envs = ()
    default_model = "qwen3:8b"
    fast_model = "qwen3:4b"
    signup_url = "https://ollama.com/download"

    def base_url(self) -> str:
        return (self._env.get("OLLAMA_BASE_URL") or "").strip() or "http://localhost:11434"

    def local_models(self) -> list[str]:
        """Names of the models already pulled into the local Ollama server."""
        import httpx

        response = httpx.get(f"{self.base_url()}/api/tags", timeout=3)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]

    def check_runtime(self, model: str | None = None) -> None:
        model = model or self.default_model
        try:
            pulled = self.local_models()
        except Exception:  # noqa: BLE001 - any failure here means "not reachable"
            raise ProviderConfigError(
                f"Can't reach Ollama at {self.base_url()}. Install it from {self.signup_url}, make sure it's "
                f"running, then run: ollama pull {model}"
            ) from None
        if model not in pulled and f"{model}:latest" not in pulled:
            raise ProviderConfigError(f"Ollama doesn't have {model!r} yet. Run: ollama pull {model}")

    def _create(self, model: str, max_tokens: int, **kwargs: Any) -> BaseChatModel:
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, base_url=self.base_url(), num_predict=max_tokens, **kwargs)
