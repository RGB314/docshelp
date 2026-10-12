"""E2E with a real model: lesson 8.10 against this repo's server with auth (`langgraph.auth.json`).

A module of its own so that it starts after test_real_model.py's open server has stopped: both serve from the
repo folder, where `langgraph dev` keeps its local persistence (`.langgraph_api/`).
"""

import pytest

from .helpers import ROOT, langgraph_server, run_lesson
from .test_real_model import ollama_env, requires_ollama  # noqa: F401 - a fixture, used by name

pytestmark = pytest.mark.e2e


@requires_ollama
def test_auth_lesson_against_the_real_auth_server(ollama_env):  # noqa: F811
    with langgraph_server(ROOT, ollama_env, startup_timeout=300, config="langgraph.auth.json") as url:
        result = run_lesson("8.10", {**ollama_env, "LANGGRAPH_AUTH_URL": url}, timeout=1500)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert "account_id=acct-1001" in result.stdout
