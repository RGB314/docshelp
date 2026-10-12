"""E2E: every lesson and stage, run as a real process with no API keys, exits cleanly with setup instructions."""

import pytest

from docshelp.tasks import find_lessons

from .helpers import PROVIDER_KEYS, base_env, free_port, run_evals, run_lesson

pytestmark = pytest.mark.e2e

LESSONS = [code for code, _ in find_lessons()]


@pytest.mark.parametrize("code", LESSONS)
def test_lesson_without_key_explains_what_to_set(code):
    result = run_lesson(code, base_env(DOCSHELP_PROVIDER="groq"))

    assert result.returncode == 1, result.stdout[-2000:] + result.stderr[-2000:]
    assert "GROQ_API_KEY" in result.stderr
    assert "Traceback" not in result.stderr


def test_evals_without_key_explain_what_to_set():
    result = run_evals(base_env(DOCSHELP_PROVIDER="anthropic"))

    assert result.returncode == 1, result.stdout[-2000:] + result.stderr[-2000:]
    assert "ANTHROPIC_API_KEY" in result.stdout + result.stderr
    assert "Traceback" not in result.stdout + result.stderr


@pytest.mark.parametrize("provider", PROVIDER_KEYS)
def test_every_keyed_provider_explains_its_missing_key(provider):
    result = run_lesson("1.1", base_env(DOCSHELP_PROVIDER=provider))

    assert result.returncode == 1
    assert PROVIDER_KEYS[provider] in result.stderr
    assert "Traceback" not in result.stderr


def test_default_provider_is_local_ollama_and_explains_when_it_is_not_running():
    env = base_env(OLLAMA_BASE_URL=f"http://127.0.0.1:{free_port()}")
    env["DOCSHELP_PROVIDER"] = ""  # set but empty, so a developer's .env can't pick another provider

    result = run_lesson("1.1", env)

    assert result.returncode == 1
    assert "Can't reach Ollama" in result.stderr
    assert "Traceback" not in result.stderr


def test_vision_lesson_names_the_vision_model_to_pull():
    env = base_env(DOCSHELP_PROVIDER="ollama", OLLAMA_BASE_URL=f"http://127.0.0.1:{free_port()}")
    result = run_lesson("1.4", env)

    assert result.returncode == 1
    assert "ollama pull" in result.stderr


def test_graph_capstone_with_ollama_not_running_explains_what_to_do():
    env = base_env(DOCSHELP_PROVIDER="ollama", OLLAMA_BASE_URL=f"http://127.0.0.1:{free_port()}")
    result = run_lesson("3.15", env)

    assert result.returncode == 1
    assert "Can't reach Ollama" in result.stderr
    assert "Traceback" not in result.stderr


def test_unknown_provider_is_rejected_with_the_options():
    result = run_lesson("1.1", base_env(DOCSHELP_PROVIDER="mistral"))
    assert result.returncode == 1
    assert "anthropic, openai, google, groq, ollama" in result.stderr


def test_unknown_lesson_lists_the_lessons():
    result = run_lesson("99.1", base_env())
    assert result.returncode == 1
    assert "1.1" in result.stderr


@pytest.mark.parametrize(("code", "var", "task"), [("8.3", "LANGGRAPH_URL", "poe server"),
                                                   ("8.10", "LANGGRAPH_AUTH_URL", "poe server-auth")])
def test_server_lessons_without_a_server_explain_how_to_start_it(code, var, task):
    env = base_env(DOCSHELP_PROVIDER="groq", GROQ_API_KEY="not-used", **{var: f"http://127.0.0.1:{free_port()}"})
    result = run_lesson(code, env, timeout=120)

    assert result.returncode == 1
    assert task in result.stderr
    assert "Traceback" not in result.stderr
