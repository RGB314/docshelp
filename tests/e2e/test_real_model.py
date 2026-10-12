"""E2E with a real model and no API keys: every lesson against a local Ollama model.

Opt-in (it needs Ollama and a pulled model), e.g.:
    ollama pull qwen2.5:0.5b
    DOCSHELP_E2E_MODEL=qwen2.5:0.5b uv run poe e2e
CI runs this inside Docker Compose with the `ollama` service.

Small models give weak answers; these tests check that everything runs to completion, not answer quality.
`DOCSHELP_E2E_PARTS=7,9` runs only those parts' lessons: CI splits the real-model run into parallel shards.
Part 8's lessons are clients, so they run against this repo's own servers (`langgraph.json`, then
`langgraph.auth.json`), started here with the real model.
"""

import os

import pytest

from docshelp.tasks import find_lessons

from .helpers import ROOT, base_env, langgraph_server, ollama_ready, run_evals, run_lesson

pytestmark = pytest.mark.e2e

MODEL = os.getenv("DOCSHELP_E2E_MODEL", "")
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

requires_ollama = pytest.mark.skipif(
    not (MODEL and ollama_ready(BASE_URL, MODEL)),
    reason="set DOCSHELP_E2E_MODEL to a model pulled into a running Ollama to run real-model E2E tests",
)


@pytest.fixture(scope="module")
def ollama_env():
    return base_env(
        DOCSHELP_PROVIDER="ollama",
        DOCSHELP_MODEL=MODEL,
        DOCSHELP_FAST_MODEL=MODEL,
        DOCSHELP_MAX_TOKENS="512",
        OLLAMA_BASE_URL=BASE_URL,
    )


PARTS = {part.strip() for part in os.getenv("DOCSHELP_E2E_PARTS", "").split(",") if part.strip()}


def selected(part: str) -> bool:
    """Is this part in the run? (No DOCSHELP_E2E_PARTS: every part.)"""
    return not PARTS or part in PARTS


LESSONS = [code for code, _ in find_lessons() if selected(code.split(".")[0])]
# Lesson 1.4 needs an image-capable model, which the tiny CI model is not. Part 8 needs a server (below).
STANDALONE_LESSONS = [code for code in LESSONS if code != "1.4" and not code.startswith("8.")]
SERVER_LESSONS = [code for code in LESSONS if code.startswith("8.") and code != "8.10"]


@requires_ollama
@pytest.mark.parametrize("code", STANDALONE_LESSONS)
def test_lesson_runs_to_completion(ollama_env, code):
    result = run_lesson(code, ollama_env, timeout=1500)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]


@requires_ollama
@pytest.mark.skipif(not selected("5"), reason="the evals belong to Part 5, which isn't in DOCSHELP_E2E_PARTS")
def test_evals_run_against_the_real_model(ollama_env):
    result = run_evals(ollama_env, timeout=1500)
    # A tiny model may fail an eval; what matters is that the suite runs and reports (exit 0 or 1, not 2+).
    assert result.returncode in (0, 1), result.stdout[-3000:] + result.stderr[-3000:]
    assert "passed" in result.stdout or "failed" in result.stdout


@pytest.fixture(scope="module")
def server_url(ollama_env):
    with langgraph_server(ROOT, ollama_env, startup_timeout=300) as url:
        yield url


@requires_ollama
@pytest.mark.parametrize("code", SERVER_LESSONS)
def test_deployment_lesson_against_the_real_server(ollama_env, server_url, code):
    result = run_lesson(code, {**ollama_env, "LANGGRAPH_URL": server_url}, timeout=1500)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
