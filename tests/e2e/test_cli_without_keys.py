"""E2E: every stage, run as a real process with no API keys, exits cleanly with setup instructions."""

import pytest

from .helpers import PROVIDER_KEYS, base_env, free_port, run_stage

pytestmark = pytest.mark.e2e

STAGES_NEEDING_A_MODEL = [
    "01_langchain_basics.py",
    "02_langchain_rag.py",
    "03_langgraph_agent.py",
    "04_langsmith_eval.py",
]


# Every stage once with the default provider, plus every other provider once (each run is a fresh process).
CASES = [(script, "anthropic") for script in STAGES_NEEDING_A_MODEL] + [
    ("01_langchain_basics.py", provider) for provider in PROVIDER_KEYS if provider != "anthropic"
]


@pytest.mark.parametrize("script, provider", CASES)
def test_stage_without_key_explains_what_to_set(script, provider):
    result = run_stage(script, base_env(DOCSHELP_PROVIDER=provider))

    assert result.returncode == 1
    assert PROVIDER_KEYS[provider] in result.stderr
    assert "Traceback" not in result.stderr


def test_stage_with_ollama_not_running_explains_what_to_do():
    env = base_env(DOCSHELP_PROVIDER="ollama", OLLAMA_BASE_URL=f"http://127.0.0.1:{free_port()}")
    result = run_stage("03_langgraph_agent.py", env)

    assert result.returncode == 1
    assert "Can't reach Ollama" in result.stderr
    assert "Traceback" not in result.stderr


def test_unknown_provider_is_rejected_with_the_options():
    result = run_stage("01_langchain_basics.py", base_env(DOCSHELP_PROVIDER="mistral"))
    assert result.returncode == 1
    assert "anthropic, openai, google, groq, ollama" in result.stderr


def test_stage5_without_a_server_explains_how_to_start_it():
    env = base_env(LANGGRAPH_URL=f"http://127.0.0.1:{free_port()}")
    result = run_stage("05_langgraph_server.py", env, timeout=60)

    assert result.returncode == 1
    assert "poe server" in result.stderr
    assert "Traceback" not in result.stderr
