"""E2E for Part 8: every deployment lesson against a real LangGraph Server (no keys, no network).

Two `langgraph dev` servers serve the repo's own `langgraph.json` and `langgraph.auth.json`, with the fakes from
`fake_server/fake_app.py` (a rule-based model, fake embeddings, data in a temp dir). Each lesson's `run()` is
driven in-process with a real SDK client, so the assertions see what the lesson returns; one lesson also runs
as a process, exactly as a learner starts it.
"""

import pytest
from langgraph_sdk import get_sync_client

from docshelp.auth import issue_token
from docshelp.tasks import load_lesson

from .helpers import base_env, fake_server_dir, langgraph_server, run_lesson

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="module")
def env(data_dir):
    # 0.3 s per model call: slow enough that a second message arrives while the first run works (8.4).
    return base_env(DOCSHELP_E2E_DATA=str(data_dir), DOCSHELP_E2E_FAKE_DELAY="0.3", DOCSHELP_PROVIDER="ollama")


@pytest.fixture(scope="module")
def url(tmp_path_factory, env):
    with langgraph_server(fake_server_dir(tmp_path_factory.mktemp("server")), env, startup_timeout=300) as server:
        yield server


@pytest.fixture(scope="module")
def auth_url(tmp_path_factory, env):
    folder = fake_server_dir(tmp_path_factory.mktemp("auth-server"), "langgraph.auth.json")
    with langgraph_server(folder, env, startup_timeout=300) as server:
        yield server


@pytest.fixture(scope="module")
def client(url):
    return get_sync_client(url=url)


def test_8_1_concepts_lists_every_graph_and_its_assistant(client, url):
    result = load_lesson("8.1").run(client, url)
    assert set(result["graphs"]) == result["assistants"]
    assert {"docshelp", "support_agent", "ops_analyst", "agent_from_template", "inbox_ingest"} <= result["assistants"]
    assert result["nodes"][:2] == ["classify", "retrieve"] or "classify" in result["nodes"]
    assert {"style", "model", "account_id"} <= set(result["context_fields"])


def test_8_2_sdk_clients_and_remote_graph(client, url):
    result = load_lesson("8.2").run(client, url)
    assert "export a notebook" in result["stateless"]
    assert "version history" in result["async"]
    assert "What's my name?" in result["remembered"]  # the fake model echoes the second turn
    assert result["escalated"] > 1  # the product question went through the deployed graph
    assert result["small_talk"] == 1  # small talk never left the local graph


def test_8_3_threads_runs_streaming_and_approval(client, data_dir):
    result = load_lesson("8.3").run(client)
    assert [t["title"] for t in result["approvals"]] == ["Webhooks not delivered"]
    assert "I've opened ticket NIM-" in result["answers"][1]
    assert "Webhooks not delivered" in (data_dir / "tickets.jsonl").read_text(encoding="utf-8")
    assert "__interrupt__" in result["nodes"] and "open_ticket" in result["nodes"]
    assert result["tokens"] > 3  # streamed word by word
    assert result["runs"] and set(result["runs"]) == {"success"}
    assert result["checkpoints"] > 5 and result["found"] >= 1 and result["fork"]


def test_8_4_double_texting_strategies(client):
    result = load_lesson("8.4").run(client)
    first, second = "Sync is broken.", "On Windows, if that matters."
    assert all(r["first_was"] == "running" for r in result.values())
    assert result["reject"]["second"] == "rejected" and result["reject"]["customer_messages"] == [first]
    assert result["enqueue"]["first"] == result["enqueue"]["second"] == "success"
    assert result["enqueue"]["customer_messages"] == [first, second]
    assert result["interrupt"]["first"] == "interrupted" and result["interrupt"]["second"] == "success"
    assert result["rollback"]["first"] == "deleted" and result["rollback"]["customer_messages"] == [second]


def test_8_5_assistants_configure_and_version_one_graph(client):
    result = load_lesson("8.5").run(client)
    assert set(result["answers"]) == {"docshelp-chat", "docshelp-email"}
    assert result["versions"] == {1: {"style": "detailed"}, 2: {"style": "detailed", "model": "fast"}}
    assert result["rolled_back_to"] == 1
    assert result["found"] == ["docshelp-chat", "docshelp-email"]
    assert not client.assistants.search(metadata={"team": "support"})  # the lesson tidies up


def test_8_6_custom_routes(client, url):
    result = load_lesson("8.6").run(client, url)
    assert result["health"]["status"] == "ok"
    assert result["tickets"] >= 1 and result["high"] >= 1  # the ticket lesson 8.3 opened
    assert result["ui"] == 200
    assert result["documented"] == ["/health", "/tickets", "/ui"] and result["api_ok"]


def test_8_7_server_store_memory_and_semantic_search(client):
    result = load_lesson("8.7").run(client)
    assert ["profile", "acct-1001"] in result["namespaces"] and ["issues", "acct-1001"] in result["namespaces"]
    assert result["profile"]["operating_systems"] == ["Windows"]  # saved by the agent via trustcall
    assert "crm-1" in result["hits"] and result["solved"] == ["crm-1"]


def test_8_8_deep_agent_memory_lives_in_the_server_store(client):
    result = load_lesson("8.8").run(client)
    assert "/reports/sync-incident.md" in result["approvals"]
    assert "TL;DR" not in result["before"] and "TL;DR" in result["after"]


def test_8_9_templates_deploy_as_assistants(client):
    result = load_lesson("8.9").run(client)
    assert result["deployed"] == ["refund-desk", "docs-researcher"]
    assert all("You asked" in answer for answer in result["answers"].values())
    assert not client.assistants.search(graph_id="agent_from_template", limit=100)[1:]  # only the default


def test_8_10_auth(auth_url):
    result = load_lesson("8.10").run(auth_url)
    assert result["no_token"] == result["expired"] == "401 unauthenticated"
    assert result["grace_reads_adas_thread"] == "404 not found" and result["grace_sees"] == 0
    assert result["staff_sees_adas_thread"]
    assert result["context_account"] == "acct-1001"  # not the acct-2002 the client asked for
    assert result["customer_creates_assistant"] == "403 forbidden"
    assert result["other_memory"] == "403 forbidden"


def test_8_11_raw_http_streaming_like_a_ui(client, url):
    result = load_lesson("8.11").run(client, url)
    assert result["events"].get("messages", 0) > 3 and result["events"].get("updates", 0) >= 2
    assert "two-factor" in result["answer"]
    assert "model" in result["steps"] and "tools" in result["steps"]


def test_8_12_ambient_agent_ingest_cron_and_inbox(client):
    result = load_lesson("8.12").run(client, emails=2)
    assert len(result["started"]) == 2 and result["again"] == []
    assert result["cron_schedule"] == "*/15 * * * *"
    assert result["waiting"]  # the fake model drafts a reply, which waits for approval
    assert result["resumed"]
    assert not client.crons.search()


def test_8_13_production_image(client, url, tmp_path):
    result = load_lesson("8.13").run(client, url, workdir=tmp_path)
    assert result["dockerfile"][0].startswith("FROM langchain/langgraph-api")
    assert any("LANGSERVE_GRAPHS" in line and "support_agent" in line for line in result["dockerfile"])
    assert {"app", "server", "server-auth"} <= set(result["services"])
    assert "support_agent" in result["graphs"]


def test_a_part8_lesson_runs_as_a_process_against_the_server(url, auth_url, env):
    token_env = {**env, "DOCSHELP_PROVIDER": "groq", "GROQ_API_KEY": "not-checked-by-the-client"}
    result = run_lesson("8.1", {**token_env, "LANGGRAPH_URL": url}, timeout=300)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert "support_agent" in result.stdout
    token = issue_token("acct-1001")
    assert token  # the lesson signs its own tokens with the same default secret as the server
    result = run_lesson("8.10", {**token_env, "LANGGRAPH_AUTH_URL": auth_url}, timeout=300)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert "account_id=acct-1001" in result.stdout
