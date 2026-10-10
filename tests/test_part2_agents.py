"""Part 2 lessons (production-ready agents), each driven offline through its `run()` with scripted models."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from docshelp import tools
from docshelp.customers import ACCOUNTS, Customer

from .conftest import FakeChatModel, tool_call

ADA = Customer(name="Ada", plan="team", account_id="acct-1001")


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    """Send tickets and outgoing emails to a temp dir instead of data/."""
    monkeypatch.setattr(tools, "TICKETS_PATH", tmp_path / "tickets.jsonl")
    monkeypatch.setattr(tools, "OUTBOX_PATH", tmp_path / "outbox.jsonl")
    return tmp_path


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def tool_texts(messages):
    return [m.text for m in messages if isinstance(m, ToolMessage)]


@tool
def service_status(region: str) -> str:
    """Current status of the Nimbus Notes service in a region."""
    return f"{region}: degraded (INC-2041)"


# --- shared app code -------------------------------------------------------------------------------


def test_context_tools_take_no_identity_arguments():
    # The account comes from runtime context, so the model can't ask about someone else's account.
    assert tools.my_account.args == {}
    assert set(tools.open_ticket.args) == {"summary", "priority"}


def test_every_account_has_a_contact_and_a_region():
    for account in ACCOUNTS.values():
        assert account.contacts and account.region


# --- lessons ------------------------------------------------------------------------------------------


def test_2_1_mcp_tools_resources_and_prompts_over_stdio(lesson, fake_retriever):
    llm = FakeChatModel(replies=[tool_call("service_status", {"region": "eu-west"}), "Sync in eu-west is degraded."])

    result = lesson("2.1").run(llm, fake_retriever)

    assert {"service_status", "list_incidents"} <= set(result["tools"])  # discovered from the server
    assert "eu-west" in result["resource"]  # the status page, read as an MCP resource
    assert isinstance(result["prompt"][0], HumanMessage) and "eu-west" in result["prompt"][0].text
    assert any("INC-2041" in t for t in tool_texts(result["messages"]))  # the server process ran the tool
    assert result["answer"] == "Sync in eu-west is degraded."


def test_2_2_context_and_custom_state(lesson):
    llm = FakeChatModel(
        replies=[
            tool_call("my_account", {}),
            tool_call("set_device", {"device": "windows"}, "c2"),
            tool_call("sync_fix", {}, "c3"),
            "Restart the app, then check the tray icon.",
        ]
    )

    result = lesson("2.2").run(llm)

    texts = tool_texts(result["state"]["messages"])
    assert "Acme Research" in texts[0]  # read from runtime context (account acct-1001)
    assert result["state"]["device"] == "windows"  # written to custom state by a tool's Command
    assert "Windows" in texts[2]  # a later tool read that state
    assert result["answer"].startswith("Restart")


def test_2_3_supervisor_delegates_to_specialists_with_context(lesson, fake_retriever):
    supervisor = FakeChatModel(
        replies=[
            tool_call("ask_status", {"question": "Is sync degraded in eu-west?"}),
            tool_call("ask_billing", {"question": "How many seats are free?"}, "c2"),
            "Sync is degraded in eu-west, and you have 1 free seat.",
        ]
    )
    specialists = FakeChatModel(
        replies=[
            tool_call("service_status", {"region": "eu-west"}),
            "eu-west is degraded (INC-2041).",
            tool_call("my_account", {}, "c2"),
            "11 of 12 seats are used.",
        ]
    )

    result = lesson("2.3").run(supervisor, fake_retriever, [service_status], specialist_llm=specialists)

    assert result["consulted"] == ["ask_status", "ask_billing"]
    assert tool_texts(result["messages"]) == ["eu-west is degraded (INC-2041).", "11 of 12 seats are used."]
    # The billing specialist got the caller's context from the supervisor: it looked up Ada's account.
    assert any("Acme Research" in m.text for m in specialists.seen[-1] if isinstance(m, ToolMessage))
    assert result["answer"].endswith("1 free seat.")


def test_2_4_middleware_hooks_run_in_order_and_guard_can_end_early(lesson, fake_retriever, tmp_path):
    llm = FakeChatModel(replies=[tool_call("search_docs", {"query": "export PDF"}), "Use File > Export."])
    log = tmp_path / "audit.jsonl"

    result = lesson("2.4").run(llm, fake_retriever, log)

    assert result["events"] == [
        "before_agent",
        "before_model", "model", "after_model",
        "tool",
        "before_model", "model", "after_model",
        "after_agent",
    ]
    logged = [record["event"] for record in lines(log)]
    assert logged[: len(result["events"])] == result["events"]  # every hook also wrote a JSON line
    assert "model" not in logged[len(result["events"]) :]  # the guarded run made no model call
    assert "password" in result["blocked"].lower()
    assert len(llm.seen) == 2  # the guarded message never reached the model


def test_2_5_prebuilt_middleware_retries_limits_and_fallback(lesson, fake_retriever):
    llm = FakeChatModel(
        replies=[
            # retries: the flaky tool fails once and succeeds on the retry
            tool_call("check_sync_service", {}),
            "The sync service is up.",
            # limits: the model keeps searching; the 3rd search is refused, the 4th model call never happens
            tool_call("search_docs", {"query": "a"}),
            tool_call("search_docs", {"query": "b"}, "c2"),
            tool_call("search_docs", {"query": "c"}, "c3"),
            # fallback: the primary model is broken, so this backup answers
            "Answer from the backup model.",
        ]
    )
    broken = FakeChatModel(replies=[RuntimeError("model not found"), RuntimeError("model not found")])

    result = lesson("2.5").run(llm, broken, fake_retriever)

    assert result["attempts"] == 2
    assert result["searches_run"] == 2
    assert len(broken.seen) == 2  # retried once, then fell back
    assert result["fallback_answer"] == "Answer from the backup model."
    assert not llm.replies  # every scripted reply used: no extra model calls


def test_2_6_long_conversations_summarize_trim_and_clear(lesson, fake_retriever):
    llm = FakeChatModel(
        replies=[
            "Hi Ada.", "Noted: Windows.", "Noted: 12 seats.", "Try restarting sync.",  # summarization
            "Hello.", "Noted.", "Sure.",  # trimming
            tool_call("search_docs", {"query": "a"}), tool_call("search_docs", {"query": "b"}, "c2"),
            tool_call("search_docs", {"query": "c"}, "c3"), "Done.",  # context editing
        ]
    )
    fast = FakeChatModel(replies=["Ada, Team plan, Windows, 12 seats, sync stuck."])

    result = lesson("2.6").run(llm, fast, fake_retriever)

    assert "Windows" in result["summary"]
    assert result["summarized_seen"][-1] < result["summarized_seen"][-2]  # older turns replaced by a summary
    assert max(result["trimmed_seen"]) <= 4
    assert any("[cleared]" in m.text for m in llm.seen[-1] if isinstance(m, ToolMessage))


def test_2_7_human_in_the_loop_approve_edit_reject(lesson, tmp_data):
    llm = FakeChatModel(
        replies=[
            tool_call("open_ticket", {"summary": "Sync stuck", "priority": "normal"}),
            "Ticket opened.",
            tool_call("open_ticket", {"summary": "Data loss after sync", "priority": "normal"}, "c2"),
            "Ticket opened as urgent.",
            tool_call("open_ticket", {"summary": "Dark mode request", "priority": "low"}, "c3"),
            "Understood, no ticket.",
        ]
    )

    result = lesson("2.7").run(llm, customer=ADA)

    assert [r["name"] for r in result["interrupts"]] == ["open_ticket"] * 3
    saved = lines(tmp_data / "tickets.jsonl")
    assert [t["summary"] for t in saved] == ["Sync stuck", "Data loss after sync"]  # the rejected one never ran
    assert saved[1]["priority"] == "urgent"  # the reviewer's edit is what executed
    assert saved[0]["account_id"] == "acct-1001"  # from runtime context
    assert result["replies"][-1] == "Understood, no ticket."


def test_2_8_dynamic_model_and_tools(lesson, fake_retriever):
    strong = FakeChatModel(replies=["Here is how to fix SSO with SAML ..."])
    fast = FakeChatModel(replies=["Use Ctrl+N.", "Use Ctrl+N."])

    result = lesson("2.8").run(strong, fast, fake_retriever)

    assert result["models"] == ["fast", "strong", "fast"]
    assert "open_ticket" not in result["tools"]["free"]
    assert "open_ticket" in result["tools"]["team"]
    assert fast.bound[0] == result["tools"]["free"]  # the model really was offered only those tools


def test_2_9_capstone_email_assistant(lesson, fake_retriever, tmp_data):
    reply = {"to": "ada@acme.example", "subject": "Re: more seats", "body": "You can add seats in Admin > Billing."}
    llm = FakeChatModel(
        replies=[
            tool_call("verify_sender", {"email": "ada@acme.example", "account_id": "acct-1001"}),
            tool_call("lookup_account", {}, "c2"),
            tool_call("search_docs", {"query": "add seats"}, "c3"),
            tool_call("send_reply", reply, "c4"),
            "Reply sent to Ada.",
        ]
    )

    result = lesson("2.9").run(llm, fake_retriever)

    # Account tools stay hidden until the sender is verified (dynamic tools from custom state).
    assert "lookup_account" not in llm.bound[0] and "lookup_account" in llm.bound[1]
    assert result["state"]["verified_account"] == "acct-1001"
    assert [r["name"] for r in result["interrupts"]] == ["send_reply"]  # a human approved the send
    assert [m["subject"] for m in lines(tmp_data / "outbox.jsonl")] == ["Re: more seats"]
    assert result["answer"] == "Reply sent to Ada."


def test_2_9_unverified_sender_gets_no_account_data(lesson, fake_retriever, tmp_data):
    llm = FakeChatModel(
        replies=[
            tool_call("verify_sender", {"email": "mallory@evil.example", "account_id": "acct-1001"}),
            "I can't share account details until you're verified.",
        ]
    )

    result = lesson("2.9").run(llm, fake_retriever)

    assert result["state"].get("verified_account") is None
    assert "lookup_account" not in llm.bound[-1]
    assert lines(tmp_data / "outbox.jsonl") == []
