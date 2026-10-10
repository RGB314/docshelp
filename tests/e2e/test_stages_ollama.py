"""E2E with a real model and no API keys: every lesson and stage against a local Ollama model.

Opt-in (it needs Ollama and a pulled model), e.g.:
    ollama pull qwen2.5:0.5b
    DOCSHELP_E2E_MODEL=qwen2.5:0.5b uv run poe e2e
CI runs this inside Docker Compose with the `ollama` service.

Small models give weak answers; these tests check that everything runs to completion, not answer quality.
"""

import os

import pytest

from docshelp.tasks import find_lessons

from .helpers import ROOT, base_env, langgraph_server, ollama_ready, run_evals, run_lesson, run_stage

pytestmark = pytest.mark.e2e

MODEL = os.getenv("DOCSHELP_E2E_MODEL", "")
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

requires_ollama = pytest.mark.skipif(
    not (MODEL and ollama_ready(BASE_URL, MODEL)),
    reason="set DOCSHELP_E2E_MODEL to a model pulled into a running Ollama to run real-model E2E tests",
)


@pytest.fixture
def ollama_env(tmp_path):
    return base_env(
        DOCSHELP_PROVIDER="ollama",
        DOCSHELP_MODEL=MODEL,
        DOCSHELP_FAST_MODEL=MODEL,
        DOCSHELP_MAX_TOKENS="512",
        OLLAMA_BASE_URL=BASE_URL,
    )


# Lesson 1.4 needs an image-capable model, which the tiny CI model is not.
TEXT_ONLY_LESSONS = [code for code, _ in find_lessons() if code != "1.4"]


@requires_ollama
@pytest.mark.parametrize("code", TEXT_ONLY_LESSONS)
def test_lesson_runs_to_completion(ollama_env, code):
    result = run_lesson(code, ollama_env, timeout=1500)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]


@requires_ollama
def test_evals_run_against_the_real_model(ollama_env):
    result = run_evals(ollama_env, timeout=1500)
    # A tiny model may fail an eval; what matters is that the suite runs and reports (exit 0 or 1, not 2+).
    assert result.returncode in (0, 1), result.stdout[-3000:] + result.stderr[-3000:]
    assert "passed" in result.stdout or "failed" in result.stdout


@requires_ollama
def test_stage5_against_the_real_server(ollama_env):
    with langgraph_server(ROOT, ollama_env, startup_timeout=300) as url:
        result = run_stage("05_langgraph_server.py", {**ollama_env, "LANGGRAPH_URL": url}, timeout=900)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert result.stdout.count("DocsHelp:") == 2
