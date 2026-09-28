"""Model providers: registry/factory, env-based configuration and per-provider behaviour."""

import pytest
from pydantic import BaseModel

from docshelp.providers import LLMProvider, ProviderConfigError, ProviderRegistry, registry
from docshelp.providers import base as provider_base

# provider name -> (API key env var, LangChain chat model class, structured-output method)
EXPECTED = {
    "anthropic": ("ANTHROPIC_API_KEY", "ChatAnthropic", "json_schema"),
    "openai": ("OPENAI_API_KEY", "ChatOpenAI", "json_schema"),
    "google": ("GOOGLE_API_KEY", "ChatGoogleGenerativeAI", "json_schema"),
    "groq": ("GROQ_API_KEY", "ChatGroq", "function_calling"),
    "ollama": (None, "ChatOllama", "json_schema"),
}
KEYED = [name for name, (key, _, _) in EXPECTED.items() if key]


@pytest.fixture(autouse=True)
def no_real_keys(monkeypatch):
    """Prove providers read keys from the env mapping they are given, never from the real environment."""
    for key, _, _ in EXPECTED.values():
        if key:
            monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def model_name(llm) -> str:
    return getattr(llm, "model", None) or getattr(llm, "model_name")


def api_key(llm) -> str:
    for field in ("anthropic_api_key", "openai_api_key", "google_api_key", "groq_api_key"):
        if (secret := getattr(llm, field, None)) is not None:
            return secret.get_secret_value()
    raise AssertionError("no api key field found")


def test_registry_lists_all_supported_providers():
    assert set(registry.names()) == set(EXPECTED)


def test_unknown_provider_error_lists_the_options():
    with pytest.raises(ProviderConfigError) as err:
        registry.create("mistral")
    for name in EXPECTED:
        assert name in str(err.value)


@pytest.mark.parametrize("name", KEYED)
def test_creates_chat_model_with_key_from_env(name):
    key_var, cls_name, _ = EXPECTED[name]
    provider = registry.create(name, env={key_var: "test-key-123"})

    llm = provider.chat_model()

    assert type(llm).__name__ == cls_name
    assert model_name(llm) == provider.default_model
    assert api_key(llm) == "test-key-123"


@pytest.mark.parametrize("name", KEYED)
def test_missing_key_error_names_env_var_and_signup_url(name):
    key_var = EXPECTED[name][0]
    provider = registry.create(name, env={})

    assert not provider.is_configured()
    with pytest.raises(ProviderConfigError) as err:
        provider.chat_model()
    assert key_var in str(err.value)
    assert provider.signup_url in str(err.value)


def test_gemini_also_accepts_gemini_api_key():
    llm = registry.create("google", env={"GEMINI_API_KEY": "g-key"}).chat_model()
    assert api_key(llm) == "g-key"


def test_ollama_needs_no_key_and_reads_base_url_from_env():
    provider = registry.create("ollama", env={"OLLAMA_BASE_URL": "http://gpu-box:11434"})
    assert provider.is_configured()
    llm = provider.chat_model()
    assert llm.base_url == "http://gpu-box:11434"


@pytest.mark.parametrize("name", list(EXPECTED))
def test_model_override_fast_model_and_max_tokens(name):
    key_var = EXPECTED[name][0]
    provider = registry.create(name, env={key_var: "k"} if key_var else {})

    assert model_name(provider.chat_model("my-custom-model")) == "my-custom-model"
    assert model_name(provider.chat_model(fast=True)) == provider.fast_model
    assert provider.fast_model != provider.default_model

    llm = provider.chat_model(max_tokens=1234)
    limit = llm.num_predict if name == "ollama" else getattr(llm, "max_tokens", None) or llm.max_output_tokens
    assert limit == 1234


@pytest.mark.parametrize("name", list(EXPECTED))
def test_structured_output_uses_provider_specific_method(name):
    calls = []

    class RecordingLLM:
        def with_structured_output(self, schema, **kwargs):
            calls.append((schema, kwargs))
            return "structured-runnable"

    class Answer(BaseModel):
        text: str

    provider = registry.create(name, env={})
    assert provider.structured(RecordingLLM(), Answer) == "structured-runnable"
    assert calls == [(Answer, {"method": EXPECTED[name][2]})]


def test_missing_integration_package_error_names_the_extra(monkeypatch):
    monkeypatch.setattr(provider_base, "find_spec", lambda _module: None)
    provider = registry.create("groq", env={"GROQ_API_KEY": "k"})

    with pytest.raises(ProviderConfigError) as err:
        provider.chat_model()
    assert "uv sync --extra groq" in str(err.value)


def test_new_providers_plug_in_without_changing_existing_code():
    """Open/closed: a new provider is just a subclass registered with a registry."""
    local_registry = ProviderRegistry()

    @local_registry.register
    class EchoProvider(LLMProvider):
        name = "echo"
        display_name = "Echo"
        extra = "echo"
        module = "json"  # any importable module
        api_key_envs = ()
        default_model = "echo-large"
        fast_model = "echo-small"
        signup_url = "https://example.com"

        def _create(self, model, max_tokens, **kwargs):
            return {"model": model, "max_tokens": max_tokens}

    assert local_registry.names() == ["echo"]
    assert local_registry.create("echo").chat_model(max_tokens=10) == {"model": "echo-large", "max_tokens": 10}


@pytest.mark.parametrize("name", KEYED)
def test_hosted_providers_have_no_runtime_check(name):
    registry.create(name, env={}).check_runtime()  # no network calls, no error


def test_ollama_runtime_check_reports_unreachable_server():
    provider = registry.create("ollama", env={"OLLAMA_BASE_URL": "http://127.0.0.1:9"})  # nothing listens here
    with pytest.raises(ProviderConfigError) as err:
        provider.check_runtime()
    assert "http://127.0.0.1:9" in str(err.value)
    assert "https://ollama.com/download" in str(err.value)


def test_ollama_runtime_check_reports_model_not_pulled(monkeypatch):
    provider = registry.create("ollama", env={})
    monkeypatch.setattr(provider, "local_models", lambda: ["llama3.2:latest"])
    with pytest.raises(ProviderConfigError, match="ollama pull qwen3:8b"):
        provider.check_runtime()
    with pytest.raises(ProviderConfigError, match="ollama pull mistral"):
        provider.check_runtime("mistral")


def test_ollama_runtime_check_accepts_pulled_models_including_latest_tag(monkeypatch):
    provider = registry.create("ollama", env={})
    monkeypatch.setattr(provider, "local_models", lambda: ["qwen3:8b", "llama3.2:latest"])
    provider.check_runtime()
    provider.check_runtime("llama3.2")


def test_registering_a_duplicate_name_is_rejected():
    local_registry = ProviderRegistry()
    local_registry.register(type(registry.create("ollama")))
    with pytest.raises(ValueError):
        local_registry.register(type(registry.create("ollama")))
