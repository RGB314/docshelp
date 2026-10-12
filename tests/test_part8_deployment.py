"""Part 8 (deployment), offline: the server-side pieces in-process, without starting a server.

The lessons themselves are SDK clients; `tests/e2e/test_part8_deployment.py` runs each one against a real
LangGraph Server. Here: auth handlers, custom routes, the deployed agent, the ingest graph and the configs.
"""

import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import jwt
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langgraph_sdk import Auth
from starlette.testclient import TestClient

from docshelp import auth, deployment, tools, webapp
from docshelp.ambient import build_inbox_agent
from docshelp.deployment import STYLE_GUIDANCE, AssistantContext, build_deployed_agent, build_inbox_ingest
from docshelp.memory import build_memory_agent

from .conftest import FakeChatModel

ROOT = Path(__file__).resolve().parents[1]
SECRET = "a-test-secret-that-is-at-least-32-bytes"


# --- auth ------------------------------------------------------------------------------------------------


def ctx(identity="acct-1001", role="customer", action="create"):
    return SimpleNamespace(user=SimpleNamespace(identity=identity), permissions=[role], action=action)


def test_tokens_round_trip_and_bad_tokens_are_refused():
    claims = auth.verify_token(auth.issue_token("acct-1001", secret=SECRET), secret=SECRET)
    assert claims["sub"] == "acct-1001" and claims["role"] == "customer"
    with pytest.raises(jwt.InvalidTokenError):
        auth.verify_token(auth.issue_token("acct-1001", secret=SECRET), secret="another-secret-of-at-least-32-bytes")
    with pytest.raises(jwt.ExpiredSignatureError):
        auth.verify_token(auth.issue_token("acct-1001", ttl_seconds=-60, secret=SECRET), secret=SECRET)
    with pytest.raises(ValueError):
        auth.issue_token("acct-nobody")


def test_authenticate_turns_a_bearer_token_into_a_user():
    user = asyncio.run(auth.authenticate(f"Bearer {auth.issue_token('support-engineer', role='staff')}"))
    assert user == {"identity": "support-engineer", "role": "staff", "permissions": ["staff"]}
    for header in (None, "", "Basic abc", "Bearer not-a-jwt"):
        with pytest.raises(Auth.exceptions.HTTPException) as err:
            asyncio.run(auth.authenticate(header))
        assert err.value.status_code == 401


def test_owners_only_stamps_and_filters_except_for_staff():
    value = {"metadata": None}
    assert asyncio.run(auth.owner_only(ctx(), value)) == {"owner": "acct-1001"}
    assert value["metadata"] == {"owner": "acct-1001"}
    assert asyncio.run(auth.owner_only(ctx("support-engineer", "staff", "read"), {})) is None


def test_only_staff_manage_assistants():
    asyncio.run(auth.assistants(ctx(action="search"), {}))
    asyncio.run(auth.assistants(ctx("ops", "staff"), {}))
    with pytest.raises(Auth.exceptions.HTTPException) as err:
        asyncio.run(auth.assistants(ctx(action="create"), {}))
    assert err.value.status_code == 403


def test_a_customers_run_always_uses_their_own_account():
    value = {"kwargs": {"context": {"name": "Ada", "account_id": "acct-2002", "plan": "enterprise"}}}
    assert asyncio.run(auth.run_as_caller(ctx(), value)) == {"owner": "acct-1001"}
    assert value["kwargs"]["context"] == {"name": "Ada", "account_id": "acct-1001", "plan": "team"}


def test_customers_only_touch_their_own_memories():
    asyncio.run(auth.own_memories(ctx(), {"namespace": ("profile", "acct-1001")}))
    asyncio.run(auth.own_memories(ctx("ops", "staff"), {"namespace": ("profile", "acct-2002")}))
    with pytest.raises(Auth.exceptions.HTTPException):
        asyncio.run(auth.own_memories(ctx(), {"namespace": ("profile", "acct-2002")}))


# --- custom routes -----------------------------------------------------------------------------------------


def test_custom_routes(tmp_path, monkeypatch):
    tickets = tmp_path / "tickets.jsonl"
    tickets.write_text(
        json.dumps({"id": "NIM-1", "priority": "high"}) + "\n" + json.dumps({"id": "NIM-2", "priority": "low"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(tools, "TICKETS_PATH", tickets)
    http = TestClient(webapp.app)
    assert http.get("/health").json() == {"status": "ok", "service": "docshelp", "tickets": 2}
    assert http.get("/tickets", params={"priority": "high"}).json()["count"] == 1
    page = http.get("/ui")
    assert page.status_code == 200 and "/runs/stream" in page.text


# --- the deployed agent, the store index and the ingest graph ------------------------------------------------


def test_the_assistant_context_sets_the_style_and_the_model(fake_retriever):
    llm, fast = FakeChatModel(replies=["Long answer."]), FakeChatModel(replies=["Short answer."])
    agent = build_deployed_agent(llm, fast, fake_retriever)
    question = {"messages": [{"role": "user", "content": "How do I share a notebook?"}]}

    agent.invoke(question, context=AssistantContext(name="Ada", plan="team", style="detailed"))
    assert STYLE_GUIDANCE["detailed"] in llm.seen[0][0].text and "Ada" in llm.seen[0][0].text
    result = agent.invoke(question, context=AssistantContext(style="concise", model="fast"))
    assert result["messages"][-1].text == "Short answer." and len(llm.seen) == 1
    assert STYLE_GUIDANCE["concise"] in fast.seen[0][0].text


def test_embed_uses_the_local_embedding_model(monkeypatch):
    monkeypatch.setattr(deployment, "_embeddings", lambda: DeterministicFakeEmbedding(size=deployment.EMBEDDING_DIMS))
    vectors = deployment.embed(["sync", "billing"])
    assert len(vectors) == 2 and len(vectors[0]) == 384


class FakeAsyncClient:
    """The calls `inbox_ingest` makes on `get_client()`, recorded."""

    def __init__(self, handled=()):
        self.handled = set(handled)
        self.threads = SimpleNamespace(create=self._create_thread)
        self.runs = SimpleNamespace(list=self._list_runs, create=self._create_run)
        self.created = []

    async def _create_thread(self, **kwargs):
        return {"thread_id": kwargs["thread_id"]}

    async def _list_runs(self, thread_id, limit=10):
        return [{"run_id": "r"}] if thread_id in self.handled else []

    async def _create_run(self, thread_id, assistant_id, input):  # noqa: A002 - the SDK's name
        self.created.append((thread_id, assistant_id, input["email"]["id"]))


def test_ingest_starts_one_run_per_new_email_and_is_idempotent(monkeypatch):
    emails = [{"id": f"msg-{n}", "subject": f"S{n}", "from": "a@b", "body": "x", "eval": {}} for n in range(3)]
    client = FakeAsyncClient(handled={deployment.email_thread_id("msg-0")})
    monkeypatch.setattr(deployment, "get_client", lambda: client)

    result = asyncio.run(build_inbox_ingest(lambda: emails).ainvoke({"limit": 2}))  # the node is async

    assert result["started"] == ["msg-1"]  # msg-0 was handled before; the limit stops at msg-1
    assert client.created == [(deployment.email_thread_id("msg-1"), "inbox_agent", "msg-1")]
    assert deployment.email_thread_id("msg-1") == deployment.email_thread_id("msg-1") != deployment.email_thread_id("msg-2")


def test_server_builds_leave_persistence_to_the_server():
    llm = FakeChatModel()
    assert build_memory_agent(llm, store=None, on_server=True).checkpointer is None
    assert build_inbox_agent(llm, hitl=True, on_server=True).checkpointer is None
    assert build_inbox_agent(llm, hitl=True).checkpointer is not None  # local runs still get one


def test_connect_explains_how_to_start_the_server():
    with pytest.raises(SystemExit) as err:
        deployment.connect("http://127.0.0.1:9", start="uv run poe server-auth")
    assert "uv run poe server-auth" in str(err.value)


# --- the configs and lesson helpers -------------------------------------------------------------------------


def test_the_auth_config_is_the_server_config_plus_auth():
    plain = json.loads((ROOT / "langgraph.json").read_text(encoding="utf-8"))
    with_auth = json.loads((ROOT / "langgraph.auth.json").read_text(encoding="utf-8"))
    assert with_auth.pop("auth") == {"path": "./src/docshelp/auth.py:auth"}
    assert with_auth == plain
    assert plain["store"]["index"]["dims"] == deployment.EMBEDDING_DIMS


def test_every_served_graph_exists_in_server_py():
    plain = json.loads((ROOT / "langgraph.json").read_text(encoding="utf-8"))
    tree = ast.parse((ROOT / "src" / "docshelp" / "server.py").read_text(encoding="utf-8"))
    names = {t.id for node in tree.body if isinstance(node, ast.Assign) for t in node.targets}
    names |= {node.name for node in tree.body if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef)}
    for spec in plain["graphs"].values():
        assert spec.startswith("./src/docshelp/server.py:") and spec.rsplit(":", 1)[1] in names


def test_lesson_8_11_parses_server_sent_events(lesson):
    lines = ["event: metadata", 'data: {"run_id": "r1"}', "", "event: messages", 'data: [{"content": "Hi"}, {}]', ""]
    events = list(lesson("8.11").read_events(SimpleNamespace(iter_lines=lambda: iter(lines))))
    assert events == [("metadata", {"run_id": "r1"}), ("messages", [{"content": "Hi"}, {}])]


def test_lesson_8_13_generates_the_production_dockerfile(lesson, tmp_path):
    dockerfile = lesson("8.13").generate_dockerfile(tmp_path / "Dockerfile")
    assert dockerfile.startswith("FROM langchain/langgraph-api")
    assert "LANGSERVE_GRAPHS" in dockerfile and "agent_from_template" in dockerfile
