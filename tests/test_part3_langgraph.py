"""Part 3 lessons (LangGraph core), each driven offline through its `run()` with scripted models."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from docshelp import tools
from docshelp.graph import Grade, Route, TicketDraft

from .conftest import FakeChatModel, ScriptedLLM, tool_call

PRICE_12 = {"plan": "team", "members": 12, "billing": "monthly"}


def test_3_1_state_nodes_and_edges(lesson):
    llm = FakeChatModel(replies=["Thanks Ada, we're looking into your sync issue."])

    state = lesson("3.1").run(llm)

    assert state["cleaned"] == "My notes won't sync since the update. Email me at [email]"
    assert state["tags"] == ["sync"]
    assert state["reply"].startswith("Thanks Ada")
    assert "[email]" in llm.seen[0][-1].text  # the model only ever saw the cleaned text


def test_3_2_conditional_edges_and_command_route_the_same(lesson):
    module = lesson("3.2")
    routes = [
        module.Route(intent="billing"),
        module.Route(intent="howto"),
        module.Route(intent="escalate"),
    ]
    llm = FakeChatModel(structured={module.Route: routes * 2})

    result = module.run(llm)

    assert result["edges"] == result["command"] == ["billing", "howto", "escalate"]


def test_3_3_react_agent_by_hand_loops_and_remembers(lesson, fake_retriever):
    llm = FakeChatModel(
        replies=[
            tool_call("quote_price", PRICE_12),
            "Team for 12 members is $96.00 per month.",
            tool_call("quote_price", {**PRICE_12, "members": 20}, "c2"),
            "For 20 members it's $160.00 per month.",
        ]
    )

    result = lesson("3.3").run(llm, fake_retriever)

    assert result["paths"] == [["assistant", "tools", "assistant"]] * 2
    assert any("12 members" in m.text for m in llm.seen[-1] if isinstance(m, AIMessage))  # turn 1 remembered
    assert result["answer"] == "For 20 members it's $160.00 per month."


def test_3_4_studio_lists_the_graphs_in_langgraph_json(lesson, fake_retriever):
    result = lesson("3.4").run(FakeChatModel(), fake_retriever)

    assert {"docshelp", "react_agent"} <= set(result["graphs"])
    assert "tools" in result["mermaid"] and "assistant" in result["mermaid"]


def test_3_5_state_schemas(lesson):
    module = lesson("3.5")
    llm = FakeChatModel(replies=["high", "high", "high"])

    result = module.run(llm)

    assert result == {"typeddict": "high", "dataclass": "high", "pydantic": "high"}
    with pytest.raises(ValidationError):  # Pydantic state validates the graph's input...
        module.pydantic_graph(llm).invoke({"message": "x", "priority": "critical"})


def test_3_6_reducers(lesson):
    llm = FakeChatModel(replies=["billing", "billing"])  # one per run of the LLM tagger

    result = lesson("3.6").run(llm)

    assert result["without_reducer"] == "InvalidUpdateError"
    assert result["with_reducer"] == ["billing", "sync"]  # both parallel branches kept, deduplicated, sorted
    assert [m.content for m in result["messages"]] == ["Hi", "Hello! (edited)"]


def test_3_7_multiple_schemas_hide_internal_state(lesson, fake_retriever):
    llm = FakeChatModel(replies=["Use File > Export > PDF."])

    result = lesson("3.7").run(llm, fake_retriever)

    assert set(result) == {"answer", "sources"}  # output schema: no documents, no draft
    assert result["answer"].startswith("Use File > Export > PDF.")
    assert result["sources"]


def test_3_8_trim_and_filter(lesson):
    llm = FakeChatModel(replies=["a", "b", "c"])

    result = lesson("3.8").run(llm)

    assert result["filter"] == {"model_saw": 3, "state_kept": 10}  # filtering leaves state alone
    assert result["delete"] == {"model_saw": 2, "state_kept": 3}  # RemoveMessage shrinks state
    assert result["trim"]["model_saw"] < 9 and isinstance(llm.seen[2][0], SystemMessage)


def test_3_9_summarizing_chatbot(lesson):
    llm = FakeChatModel(
        replies=["Hi Ada!", "Noted.", "OK.", "Try Sync now.", "Ada, Team plan, Windows, sync stuck.", "Sure."]
    )

    result = lesson("3.9").run(llm)

    assert result["summary"] == "Ada, Team plan, Windows, sync stuck."
    assert result["sizes"][3] == 2  # after the 4th turn: summarized, only the last 2 messages kept
    assert "Ada, Team plan" in llm.seen[-1][0].text  # the summary is in the next system prompt


def test_3_10_sqlite_memory_survives_a_restart(lesson, tmp_path):
    llm = FakeChatModel(replies=["Hi Ada!", "Noted.", "You're Ada, on Windows."])
    db = tmp_path / "checkpoints.sqlite"

    result = lesson("3.10").run(llm, db)

    assert db.exists()
    assert result["restored_messages"] == 4  # the second "process" loaded both earlier turns from disk
    assert any("I'm Ada" in m.text for m in llm.seen[-1] if isinstance(m, HumanMessage))
    assert result["checkpoints"] > 2


def test_3_11_stream_modes(lesson, fake_retriever):
    llm = FakeChatModel(
        replies=[tool_call("search_docs", {"query": "offline"}), "Yes, offline works on desktop."] * 3
    )

    result = lesson("3.11").run(llm, fake_retriever)

    assert result["values"] == [1, 2, 3, 4]  # message count in each full-state snapshot
    assert result["updates"] == ["assistant", "tools", "assistant"]
    assert "".join(result["tokens"]) == "Yes, offline works on desktop."
    assert "on_chat_model_stream" in result["events"]


def test_3_12_breakpoint_edit_and_answer_for_a_tool(lesson, fake_retriever):
    llm = FakeChatModel(
        replies=[
            tool_call("quote_price", PRICE_12),
            "15 members on Team: $120.00 per month.",
            tool_call("quote_price", PRICE_12, "c2"),
            "Sorry, I can't quote that right now.",
        ]
    )

    result = lesson("3.12").run(llm, fake_retriever)

    assert result["paused_at"] == ("tools",)
    assert "$120.00" in result["edited_tool_result"]  # the edited call (15 members) is what ran
    assert result["answered_for_tool"] == "Pricing is down for maintenance."  # we answered as the tools node


def test_3_13_dynamic_interrupts_validate_and_approve(lesson):
    module = lesson("3.13")
    request = module.RefundRequest(amount=240.0, reason="Charged twice", account_id=None)
    llm = FakeChatModel(structured={module.RefundRequest: [request]})

    result = module.run(llm)

    asked = [p["question"] for p in result["interrupts"]]
    assert asked[0] == asked[1]  # an invalid account id was asked for again
    assert "approve" in asked[2].lower()
    assert result["state"]["account_id"] == "acct-1001"
    assert result["state"]["status"] == "refund approved"


def test_3_14_time_travel_replay_and_fork(lesson, fake_retriever):
    llm = FakeChatModel(
        replies=[
            tool_call("quote_price", PRICE_12), "Team, 12 members: $96.00 per month.",  # original run
            tool_call("quote_price", PRICE_12, "c2"), "Team, 12 members: $96.00 per month.",  # replay
            tool_call("quote_price", {**PRICE_12, "plan": "enterprise"}, "c3"), "Enterprise is custom.",  # fork
        ]
    )

    result = lesson("3.14").run(llm, fake_retriever)

    assert result["snapshots"] >= 5
    assert result["replayed"] == result["original"]
    assert result["forked"] == "Enterprise is custom."
    assert "Enterprise" in llm.seen[-2][-1].text  # the fork really changed the question


def test_3_15_capstone_support_graph(lesson, fake_retriever, tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "TICKETS_PATH", tmp_path / "tickets.jsonl")
    llm = ScriptedLLM(
        structured={
            Route: [
                Route(intent="docs_question", standalone_question="How long is trash kept?"),
                Route(intent="ticket_request", standalone_question="Sync broken, open a ticket"),
            ],
            Grade: [Grade(relevant=True, reason="ok")],
            TicketDraft: [TicketDraft(title="Sync broken", description="Red slash", priority="high")],
        },
        text=["30 days [07-troubleshooting.md]"],
    )

    result = lesson("3.15").run(
        llm, fake_retriever, conversation=["How long is trash kept?", "Sync is broken, open a ticket"],
        diagram=tmp_path / "graph.mmd",
    )

    assert result["paths"][0] == ["classify", "retrieve", "grade", "generate"]
    assert result["paths"][1][-1] == "open_ticket"  # approved by the scripted reviewer
    assert (tmp_path / "tickets.jsonl").exists() and (tmp_path / "graph.mmd").exists()
    assert result["snapshots"] > 5


def test_capstone_retrieve_node_has_a_retry_policy():
    from langgraph.checkpoint.memory import InMemorySaver

    from docshelp.graph import build_graph

    graph = build_graph(ScriptedLLM(), retriever=object(), checkpointer=InMemorySaver())
    assert graph.builder.nodes["retrieve"].retry_policy is not None
