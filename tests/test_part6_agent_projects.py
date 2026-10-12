"""Part 6 lessons (ambient inbox agent and deep research), driven offline with scripted models."""

import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from docshelp import observability, tools
from docshelp.ambient import RouterSchema, UpdatedPreferences

from .conftest import FakeChatModel, tool_call


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    """Tickets, sent mail and traces go to a temp dir; LangSmith stays off."""
    monkeypatch.setattr(tools, "TICKETS_PATH", tmp_path / "tickets.jsonl")
    monkeypatch.setattr(tools, "OUTBOX_PATH", tmp_path / "outbox.jsonl")
    monkeypatch.setattr(observability, "TRACES_PATH", tmp_path / "traces.jsonl")
    for var in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def route(*labels):
    return {RouterSchema: [RouterSchema(reasoning="scripted", classification=label) for label in labels]}


REPLY = {"to": "ada@acme.example", "subject": "Re: Red slash", "body": "Known 4.12 issue; sync still works."}


def test_6_1_ambient_agent_triages_and_answers(lesson, fake_retriever, sandbox):
    llm = FakeChatModel(
        replies=[
            tool_call("search_changelog", {"query": "red slash sleep windows"}),
            tool_call("send_reply", REPLY, "c2"),
            tool_call("done", {}, "c3"),
        ],
        structured=route("respond", "ignore", "notify"),
    )

    results = lesson("6.1").run(llm, fake_retriever, email_ids=["msg-001", "msg-002", "msg-005"])

    assert {i: r["classification"] for i, r in results.items()} == {
        "msg-001": "respond", "msg-002": "ignore", "msg-005": "notify"
    }
    assert results["msg-001"]["tools"] == ["search_changelog", "send_reply", "done"]
    assert results["msg-002"]["tools"] == [] and results["msg-005"]["tools"] == []  # no model calls wasted
    assert [m["to"] for m in jsonl(sandbox / "outbox.jsonl")] == ["ada@acme.example"]


def test_6_2_evaluations_score_triage_tools_and_criteria(lesson, fake_retriever):
    # The inbox in file order: 001 respond, 002 ignore, 003 respond, 004 respond, 005 notify, 006 respond,
    # 007 ignore, 008 notify. The scripted triage gets one wrong (008 -> ignore).
    labels = ["respond", "ignore", "respond", "respond", "notify", "respond", "ignore", "ignore"]
    quote = {"plan": "team", "members": 15, "billing": "annual"}
    llm = FakeChatModel(
        replies=[  # the response experiment, only for msg-006 ("Quote for 15 seats")
            tool_call("quote_price", quote),
            tool_call("send_reply", {"to": "ada@acme.example", "subject": "Re: Quote", "body": "$1,152.00/year"}, "c2"),
            tool_call("done", {}, "c3"),
        ],
        structured={
            **route(*labels, "respond"),  # 8 for the triage experiment, then 1 inside the agent for msg-006
            "score": [{"score": True, "reasoning": "quoted the exact price"}],
        },
    )

    result = lesson("6.2").run(llm, fake_retriever, respond_ids=["msg-006"])

    assert result["summary"]["triage_correct"] == pytest.approx(7 / 8)
    assert result["summary"]["expected_tools"] == 1.0 and result["summary"]["meets_criteria"] == 1.0


def test_6_3_agent_inbox_accept_edit_respond_ignore(lesson, fake_retriever, sandbox):
    draft = {"to": "ada@acme.example", "subject": "Re: Red slash", "body": "Known issue, sync works."}
    better = {**draft, "body": "Known issue, sync works. Status: status.nimbusnotes.example"}
    llm = FakeChatModel(
        replies=[
            # msg-001: the first draft gets feedback, the second is accepted
            tool_call("send_reply", draft), tool_call("send_reply", better, "c2"), tool_call("done", {}, "c3"),
            # msg-003: the ticket's priority is edited to urgent, then the reply is accepted
            tool_call("open_ticket", {"summary": "SCIM duplicates", "priority": "high"}, "c4"),
            tool_call("send_reply", {"to": "grace@globex.example", "subject": "Re: SCIM", "body": "On it."}, "c5"),
            tool_call("done", {}, "c6"),
        ],
        structured=route("respond", "respond", "notify"),  # inbox order: 001, 003, 005
    )

    results = lesson("6.3").run(llm, fake_retriever)

    assert results["msg-005"]["requests"] == ["Email needs your attention (notify)"]  # ignored by the lead
    assert results["msg-001"]["requests"] == ["send_reply", "send_reply"]
    sent = jsonl(sandbox / "outbox.jsonl")
    assert [m["body"] for m in sent] == [better["body"], "On it."]  # the rejected draft was never sent
    tickets = jsonl(sandbox / "tickets.jsonl")
    assert tickets[0]["priority"] == "urgent" and tickets[0]["account_id"] == "acct-2002"  # edited; sender's account


def test_6_4_memory_learns_from_edits_and_ignores(lesson, fake_retriever):
    learned_style = "Friendly and concise. Always end with 'Cheers, The Nimbus Notes team'."
    learned_triage = "respond: customer questions. ignore: newsletters (never reply to them)."
    reply = {"to": "ada@acme.example", "subject": "Re", "body": "Known issue."}
    llm = FakeChatModel(
        replies=[
            tool_call("send_reply", reply), tool_call("done", {}, "c2"),  # msg-001: edited by the lead
            tool_call("send_reply", {**reply, "to": "news@saasweekly.example"}, "c3"),  # msg-002: ignored
            tool_call("send_reply", {**reply, "to": "sam@student.example", "body": "Use Export. Cheers"}, "c4"),
            tool_call("done", {}, "c5"),  # msg-004: accepted as is, already in the learned style
        ],
        structured={
            **route("respond", "respond", "respond"),
            UpdatedPreferences: [
                UpdatedPreferences(reasoning="adds a sign-off", preferences=learned_style),
                UpdatedPreferences(reasoning="newsletters", preferences=learned_triage),
            ],
        },
    )

    result = lesson("6.4").run(llm, fake_retriever)

    assert result["preferences"] == {"triage": learned_triage, "response": learned_style}
    assert result["results"]["msg-002"]["requests"] == ["send_reply"]  # the ignored reply
    last_system_prompt = llm.seen[-2][0].text  # msg-004's drafting call
    assert "Cheers, The Nimbus Notes team" in last_system_prompt  # the learned style is in the prompt


# --- deep research --------------------------------------------------------------------------------


def test_6_5_scoping_asks_then_writes_a_brief(lesson):
    from docshelp.deep_research import ClarifyWithUser, ResearchBrief

    llm = FakeChatModel(
        structured={
            ClarifyWithUser: [
                ClarifyWithUser(need_clarification=True, question="Who is it for, and in what format?"),
                ClarifyWithUser(need_clarification=False, verification="Got it: a one-page offline comparison."),
            ],
            ResearchBrief: [ResearchBrief(research_brief="Compare offline support across desktop, mobile, web.")],
        }
    )

    result = lesson("6.5").run(llm)

    assert result["questions"] == ["Who is it for, and in what format?"]
    assert result["brief"].startswith("Compare offline support")


def test_6_6_researcher_searches_thinks_and_compresses(lesson, fake_retriever):
    llm = FakeChatModel(
        replies=[
            tool_call("search_sources", {"query": "offline desktop mobile"}),
            tool_call("think", {"reflection": "Have desktop and mobile; check the web app."}, "c2"),
            tool_call("search_sources", {"query": "web app offline"}, "c3"),
            "Enough.",
            "Offline works on desktop and mobile, not web [03-sync-and-offline.md].",
        ]
    )

    result = lesson("6.6").run(llm, fake_retriever)

    assert result["steps"] == ["search_sources", "think", "search_sources"]
    assert result["compressed_research"].startswith("Offline works")
    assert "changelog.md" in result["raw_notes"][0] or "[source:" in result["raw_notes"][0]
    assert "Reflection recorded" not in result["raw_notes"][0]  # think results aren't research notes


def test_6_7_researcher_reads_docs_through_the_mcp_server(lesson):
    llm = FakeChatModel(
        replies=[
            tool_call("list_docs", {}),
            tool_call("read_doc", {"name": "05-import-export.md"}, "c2"),
            "Enough.",
            "Import from Evernote and Markdown; export as PDF [05-import-export.md].",
        ]
    )

    result = lesson("6.7").run(llm)  # starts the real server process over stdio

    assert "05-import-export.md" in result["raw_notes"][0]  # list_docs output, from the server
    assert "Import" in result["raw_notes"][0]  # the file's real content
    assert result["compressed_research"].startswith("Import from Evernote")


def test_docs_mcp_server_is_read_only_and_path_safe():
    from docshelp.docs_mcp_server import grep_docs, list_docs, read_doc

    assert "03-sync-and-offline.md" in list_docs()
    assert read_doc("../../pyproject.toml").startswith("No such doc")
    assert read_doc("..\\..\\.env").startswith("No such doc")
    assert "offline" in grep_docs("OFFLINE").lower()


def conduct(*topics, think_first=True):
    reflection = {"name": "think", "args": {"reflection": "split by platform"}, "id": "t0", "type": "tool_call"}
    calls = [reflection] if think_first else []
    calls += [
        {"name": "ConductResearch", "args": {"research_topic": t}, "id": f"r{n}", "type": "tool_call"}
        for n, t in enumerate(topics)
    ]
    return AIMessage("", tool_calls=calls)


COMPLETE = AIMessage("", tool_calls=[{"name": "ResearchComplete", "args": {}, "id": "done", "type": "tool_call"}])
FINDING = "Offline works on desktop and mobile [03-sync-and-offline.md]."


def test_6_8_supervisor_runs_researchers_in_parallel_within_limits(lesson, fake_retriever):
    supervisor = FakeChatModel(replies=[conduct("desktop", "mobile", "web"), COMPLETE])
    # Researchers run concurrently, so they share one reply that works for any step: an answer with no tool
    # calls (the loop ends) that is also the compressed finding.
    researchers = FakeChatModel(replies=[FINDING] * 4)

    result = lesson("6.8").run(supervisor, fake_retriever, researcher_llm=researchers, max_concurrent=2)

    assert result["turns"] == [["think", "ConductResearch", "ConductResearch", "ConductResearch"], ["ResearchComplete"]]
    assert result["notes"] == [FINDING, FINDING]  # two ran; the third was over the limit
    skipped = [m for m in result["messages"] if isinstance(m, ToolMessage) and m.text.startswith("Not run")]
    assert len(skipped) == 1


def test_6_8_malformed_tool_calls_get_an_error_reply_instead_of_crashing(lesson, fake_retriever):
    # Small models sometimes call a tool without its required arguments (seen with a 0.5B model in CI).
    bad = AIMessage(
        "",
        tool_calls=[
            {"name": "ConductResearch", "args": {}, "id": "c1", "type": "tool_call"},
            {"name": "think", "args": {}, "id": "c2", "type": "tool_call"},
        ],
    )
    supervisor = FakeChatModel(replies=[bad, COMPLETE])

    result = lesson("6.8").run(supervisor, fake_retriever, researcher_llm=FakeChatModel(), max_concurrent=2)

    errors = [m.text for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(errors) == 2 and all(e.startswith("Error:") for e in errors)
    assert result["notes"] == []  # no researcher ran


def test_6_9_capstone_deep_research_from_request_to_graded_report(lesson, fake_retriever, tmp_path):
    from docshelp.deep_research import ClarifyWithUser, ResearchBrief

    llm = FakeChatModel(
        replies=[
            conduct("Most common sync problems and their fixes", think_first=False),  # supervisor turn 1
            FINDING,  # the researcher answers without tools...
            FINDING,  # ...and compresses
            COMPLETE,  # supervisor turn 2
            "# Onboarding report\n\nSync issues... [03-sync-and-offline.md]\n\n## Sources\n- 03-sync-and-offline.md",
        ],
        structured={
            ClarifyWithUser: [
                ClarifyWithUser(need_clarification=True, question="Which topics, and how long?"),
                ClarifyWithUser(need_clarification=False, verification="Starting research."),
            ],
            ResearchBrief: [ResearchBrief(research_brief="Common customer struggles for new support hires")],
            "score": [{"score": 0.8, "reasoning": "covers the brief, cites sources"}],
        },
    )

    result = lesson("6.9").run(llm, fake_retriever, reports_dir=tmp_path)

    assert result["report"].startswith("# Onboarding report")
    assert result["path"].read_text(encoding="utf-8") == result["report"]
    assert result["grade"]["score"] == 0.8
    assert FINDING in llm.seen[-1][-1].text  # the writer worked from the researchers' findings


def test_researcher_stops_at_its_tool_budget(fake_retriever):
    from docshelp.deep_research import build_researcher, make_research_tools, research_input

    calls = [tool_call("search_sources", {"query": f"q{n}"}, f"c{n}") for n in range(10)]
    llm = FakeChatModel(replies=[*calls[:3], "summary"])  # wants to search forever

    result = build_researcher(llm, make_research_tools(fake_retriever), max_tool_rounds=2).invoke(research_input("t"))

    assert result["compressed_research"] == "summary"
    before_prompt = llm.seen[-1][-2]  # compression ran on a history with no unanswered tool call
    assert not (isinstance(before_prompt, AIMessage) and before_prompt.tool_calls)
