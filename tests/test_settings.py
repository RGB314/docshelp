"""Settings come from environment variables; config helpers use the selected provider."""

import pytest
from pydantic import BaseModel

from docshelp import config
from docshelp.config import Settings
from docshelp.providers import ProviderConfigError


def test_defaults_when_env_is_empty():
    s = Settings.from_env({})
    assert s.provider == "anthropic"
    assert s.model is None
    assert s.fast_model is None
    assert s.max_tokens == 4096
    assert s.embedding_model == "BAAI/bge-small-en-v1.5"


def test_values_are_read_and_normalised_from_env():
    s = Settings.from_env(
        {
            "DOCSHELP_PROVIDER": "  Ollama ",
            "DOCSHELP_MODEL": "llama3.2",
            "DOCSHELP_FAST_MODEL": "llama3.2:1b",
            "DOCSHELP_MAX_TOKENS": "2048",
            "DOCSHELP_EMBEDDING_MODEL": "BAAI/bge-base-en-v1.5",
        }
    )
    assert s == Settings(
        provider="ollama",
        model="llama3.2",
        fast_model="llama3.2:1b",
        max_tokens=2048,
        embedding_model="BAAI/bge-base-en-v1.5",
    )


def test_blank_values_fall_back_to_defaults():
    s = Settings.from_env({"DOCSHELP_PROVIDER": "", "DOCSHELP_MODEL": "  "})
    assert s.provider == "anthropic"
    assert s.model is None


def test_invalid_max_tokens_is_a_clear_error():
    with pytest.raises(ProviderConfigError, match="DOCSHELP_MAX_TOKENS"):
        Settings.from_env({"DOCSHELP_MAX_TOKENS": "lots"})


def test_settings_are_immutable():
    with pytest.raises(Exception):
        Settings().provider = "openai"


def test_get_llm_uses_selected_provider_and_model():
    llm = config.get_llm(settings=Settings(provider="ollama", model="llama3.2"), env={})
    assert type(llm).__name__ == "ChatOllama"
    assert llm.model == "llama3.2"


def test_get_llm_fast_uses_fast_model_override_or_provider_default():
    s = Settings(provider="ollama", fast_model="tiny")
    assert config.get_llm(fast=True, settings=s, env={}).model == "tiny"
    s = Settings(provider="ollama")
    assert config.get_llm(fast=True, settings=s, env={}).model == config.get_provider(s, env={}).fast_model


def test_get_llm_passes_max_tokens_from_settings():
    llm = config.get_llm(settings=Settings(provider="ollama", max_tokens=321), env={})
    assert llm.num_predict == 321


def test_ensure_llm_ready_exits_with_setup_instructions():
    with pytest.raises(SystemExit) as err:
        config.ensure_llm_ready(settings=Settings(provider="openai"), env={})
    assert "OPENAI_API_KEY" in str(err.value)
    assert ".env" in str(err.value)


def test_ensure_llm_ready_runs_the_runtime_check():
    with pytest.raises(SystemExit, match="Can't reach Ollama"):
        config.ensure_llm_ready(settings=Settings(provider="ollama"), env={"OLLAMA_BASE_URL": "http://127.0.0.1:9"})


def test_ensure_llm_ready_passes_when_configured():
    config.ensure_llm_ready(settings=Settings(provider="groq"), env={"GROQ_API_KEY": "k"})


def test_structured_llm_delegates_to_provider_strategy():
    class RecordingLLM:
        def with_structured_output(self, schema, **kwargs):
            return kwargs["method"]

    class Answer(BaseModel):
        text: str

    groq = Settings(provider="groq")
    assert config.structured_llm(RecordingLLM(), Answer, settings=groq) == "function_calling"
    assert config.structured_llm(RecordingLLM(), Answer, settings=Settings()) == "json_schema"


def test_describe_reports_provider_and_model():
    s = Settings(provider="ollama", model="llama3.2")
    assert config.describe(s, env={}) == "Ollama (llama3.2)"
