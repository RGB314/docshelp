"""E2E with a real model and no API keys: all five stages against a local Ollama model.

Opt-in (it needs Ollama and a pulled model), e.g.:
    ollama pull qwen2.5:0.5b
    DOCSHELP_E2E_MODEL=qwen2.5:0.5b uv run poe e2e
CI runs this inside Docker Compose with the `ollama` service.

Small models give weak answers; these tests check that every stage runs to completion, not answer quality.
"""

import os

import pytest

from .helpers import ROOT, base_env, langgraph_server, ollama_ready, run_stage

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


@requires_ollama
@pytest.mark.parametrize(
    "script, args, expected",
    [
        ("01_langchain_basics.py", (), "6. Swap the model"),
        ("02_langchain_rag.py", (), "4. Tool-calling agent"),
        ("03_langgraph_agent.py", ("--auto-approve",), "snapshots for this thread"),
        ("04_langsmith_eval.py", (), "langgraph-agent:"),
    ],
    ids=["stage1", "stage2", "stage3", "stage4"],
)
def test_stage_runs_to_completion(ollama_env, script, args, expected):
    result = run_stage(script, ollama_env, *args, timeout=1500)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert expected in result.stdout
    assert "Using Ollama" in result.stdout


@requires_ollama
def test_stage5_against_the_real_server(ollama_env):
    with langgraph_server(ROOT, ollama_env, startup_timeout=300) as url:
        result = run_stage("05_langgraph_server.py", {**ollama_env, "LANGGRAPH_URL": url}, timeout=900)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert result.stdout.count("DocsHelp:") == 2
