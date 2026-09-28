import json

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from docshelp import tools
from docshelp.graph import Grade, Route, TicketDraft, build_graph

from .conftest import ScriptedLLM


def run(graph, text, thread="t1"):
    return graph.invoke({"messages": [HumanMessage(text)]}, {"configurable": {"thread_id": thread}})


def path(graph, text, thread="t1"):
    """Return the sequence of nodes executed for one turn."""
    config = {"configurable": {"thread_id": thread}}
    return [
        node
        for chunk in graph.stream({"messages": [HumanMessage(text)]}, config, stream_mode="updates")
        for node in chunk
    ]


def test_docs_question_happy_path(fake_retriever):
    llm = ScriptedLLM(
        structured={
            Route: [Route(intent="docs_question", standalone_question="How long is trash kept?")],
            Grade: [Grade(relevant=True, reason="ok")],
        },
        text=["30 days [07-troubleshooting.md]"],
    )
    graph = build_graph(llm, fake_retriever, InMemorySaver())
    assert path(graph, "How long is trash kept?") == ["classify", "retrieve", "grade", "generate"]
    state = graph.get_state({"configurable": {"thread_id": "t1"}}).values
    assert state["messages"][-1].content == "30 days [07-troubleshooting.md]"
    assert len(state["documents"]) == 3


def test_irrelevant_docs_trigger_one_rewrite_then_answer(fake_retriever):
    llm = ScriptedLLM(
        structured={
            Route: [Route(intent="docs_question", standalone_question="q")],
            Grade: [Grade(relevant=False, reason="no"), Grade(relevant=False, reason="still no")],
        },
        text=["better query", "I don't know."],
    )
    graph = build_graph(llm, fake_retriever, InMemorySaver())
    assert path(graph, "q") == [
        "classify", "retrieve", "grade", "rewrite", "retrieve", "grade", "generate",
    ]


def test_memory_across_turns(fake_retriever):
    llm = ScriptedLLM(
        structured={Route: [Route(intent="small_talk", standalone_question="hi")] * 2},
        text=["Hello!", "You're welcome!"],
    )
    graph = build_graph(llm, fake_retriever, InMemorySaver())
    run(graph, "hi")
    state = run(graph, "thanks")
    assert [m.content for m in state["messages"]] == ["hi", "Hello!", "thanks", "You're welcome!"]


@pytest.fixture
def ticket_llm():
    return ScriptedLLM(
        structured={
            Route: [Route(intent="ticket_request", standalone_question="open a ticket")],
            TicketDraft: [TicketDraft(title="Sync broken", description="Red slash", priority="high")],
        }
    )


def test_ticket_is_opened_only_after_human_approval(ticket_llm, fake_retriever, tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "TICKETS_PATH", tmp_path / "tickets.jsonl")
    graph = build_graph(ticket_llm, fake_retriever, InMemorySaver())
    config = {"configurable": {"thread_id": "t1"}}

    result = run(graph, "Sync is broken, open a ticket please")
    assert result["__interrupt__"][0].value["ticket"]["title"] == "Sync broken"
    assert graph.get_state(config).next == ("human_review",)
    assert not (tmp_path / "tickets.jsonl").exists()

    result = graph.invoke(Command(resume={"action": "approve"}), config)
    assert "I've opened ticket NIM-" in result["messages"][-1].content
    saved = json.loads((tmp_path / "tickets.jsonl").read_text())
    assert saved["priority"] == "high"


def test_ticket_rejected(ticket_llm, fake_retriever, tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "TICKETS_PATH", tmp_path / "tickets.jsonl")
    graph = build_graph(ticket_llm, fake_retriever, InMemorySaver())
    run(graph, "open a ticket")
    result = graph.invoke(Command(resume={"action": "reject"}), {"configurable": {"thread_id": "t1"}})
    assert "won't open a ticket" in result["messages"][-1].content
    assert not (tmp_path / "tickets.jsonl").exists()
