"""E2E: real LangGraph Server + the real stage 5 SDK client, with a scripted model (no keys, no network)."""

import shutil
from pathlib import Path

import pytest

from .helpers import base_env, langgraph_server, run_stage

pytestmark = pytest.mark.e2e

FAKE_SERVER = Path(__file__).parent / "fake_server"


def test_stage5_client_round_trip_including_human_approval(tmp_path):
    shutil.copytree(FAKE_SERVER, tmp_path / "server")
    env = base_env(DOCSHELP_E2E_TICKETS=str(tmp_path / "tickets.jsonl"))

    with langgraph_server(tmp_path / "server", env) as url:
        result = run_stage("05_langgraph_server.py", {**env, "LANGGRAPH_URL": url}, timeout=120)

    assert result.returncode == 0, result.stderr
    out = result.stdout
    # Turn 1: docs question through classify → retrieve → grade → generate
    assert "600 requests per minute [08-api.md]" in out
    # Turn 2: the run pauses at the interrupt, the client approves through the API, the ticket is opened
    assert "Ticket awaiting approval" in out
    assert "I've opened ticket NIM-" in out
    assert (tmp_path / "tickets.jsonl").read_text(encoding="utf-8").count("Webhooks not delivered") == 1
