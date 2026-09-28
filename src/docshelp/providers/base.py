"""The provider abstraction and the registry that creates providers by name.

Patterns used:
  * Strategy: each `LLMProvider` subclass encapsulates how to build and use one vendor's chat
    model (key lookup, constructor arguments, structured-output method). The rest of the app only
    talks to the `LLMProvider` interface, so vendors are interchangeable.
  * Template Method: `LLMProvider.chat_model()` fixes the algorithm (check setup, pick the model,
    build it); subclasses fill in only the vendor-specific `_create()` step.
  * Registry + Factory: providers register themselves with `@registry.register`, and
    `registry.create("groq")` builds one by name. Adding a provider means adding a class; no
    existing code changes (open/closed principle).
  * Dependency injection: providers receive the environment mapping they read from, which keeps
    them testable and means keys only ever come from environment variables (or .env).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from importlib.util import find_spec
from typing import Any, ClassVar

from langchain_core.language_models import BaseChatModel


class ProviderConfigError(RuntimeError):
    """A provider is unknown, not installed, or missing its API key. The message says how to fix it."""


class LLMProvider(ABC):
    name: ClassVar[str]  # registry key, used in DOCSHELP_PROVIDER
    display_name: ClassVar[str]
    extra: ClassVar[str]  # pyproject extra that installs the integration: `uv sync --extra <extra>`
    module: ClassVar[str]  # the LangChain integration package to import
    api_key_envs: ClassVar[tuple[str, ...]]  # env vars checked in order; empty if no key is needed
    default_model: ClassVar[str]
    fast_model: ClassVar[str]  # a smaller / cheaper model from the same provider
    signup_url: ClassVar[str]  # where to get an API key (or download the runtime)
    structured_output_method: ClassVar[str] = "json_schema"

    def __init__(self, env: Mapping[str, str] | None = None):
        self._env = os.environ if env is None else env

    # --- configuration --------------------------------------------------------------------

    def api_key(self) -> str | None:
        for var in self.api_key_envs:
            if value := (self._env.get(var) or "").strip():
                return value
        return None

    def is_installed(self) -> bool:
        return find_spec(self.module) is not None

    def is_configured(self) -> bool:
        return self.is_installed() and (not self.api_key_envs or self.api_key() is not None)

    def check(self) -> None:
        """Raise `ProviderConfigError` with fix-it instructions if this provider can't be used yet."""
        if not self.is_installed():
            raise ProviderConfigError(
                f"The {self.display_name} integration ({self.module}) is not installed. "
                f"Run: uv sync --extra {self.extra}"
            )
        if self.api_key_envs and self.api_key() is None:
            raise ProviderConfigError(
                f"{self.api_key_envs[0]} is not set. Get a key at {self.signup_url} and add it to "
                f"your .env file (see .env.example) or export it as an environment variable."
            )

    def check_runtime(self, model: str | None = None) -> None:
        """Hook for providers whose models need a running service. Hosted APIs need nothing here."""

    # --- template method ------------------------------------------------------------------

    def chat_model(
        self, model: str | None = None, *, fast: bool = False, max_tokens: int = 4096, **kwargs: Any
    ) -> BaseChatModel:
        """Build a LangChain chat model for this provider."""
        self.check()
        chosen = model or (self.fast_model if fast else self.default_model)
        return self._create(chosen, max_tokens, **kwargs)

    @abstractmethod
    def _create(self, model: str, max_tokens: int, **kwargs: Any) -> BaseChatModel:
        """Vendor-specific construction. Import the integration lazily: it's an optional extra."""

    # --- strategy hooks used by the app ---------------------------------------------------

    def structured(self, llm: BaseChatModel, schema: type) -> Any:
        """Wrap `llm` so it returns instances of `schema`, using the method this vendor supports best."""
        return llm.with_structured_output(schema, method=self.structured_output_method)

    def describe(self, model: str | None = None) -> str:
        return f"{self.display_name} ({model or self.default_model})"


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, type[LLMProvider]] = {}

    def register(self, cls: type[LLMProvider]) -> type[LLMProvider]:
        """Class decorator: `@registry.register`."""
        if cls.name in self._providers:
            raise ValueError(f"Provider {cls.name!r} is already registered")
        self._providers[cls.name] = cls
        return cls

    def names(self) -> list[str]:
        return list(self._providers)

    def create(self, name: str, env: Mapping[str, str] | None = None) -> LLMProvider:
        try:
            cls = self._providers[name.strip().lower()]
        except KeyError:
            raise ProviderConfigError(
                f"Unknown provider {name!r}. Set DOCSHELP_PROVIDER to one of: {', '.join(self.names())}"
            ) from None
        return cls(env)


registry = ProviderRegistry()
