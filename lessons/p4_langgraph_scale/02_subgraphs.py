"""Subgraphs: a ticket-review graph made of two independent graphs, run in parallel.

Academy: Intro to LangGraph M4 L2 "Sub-graphs".

Concepts:
  * a compiled graph can be a node in another graph (`builder.add_node("failures", failure_graph)`)
  * communication through *shared keys*: both subgraphs read the parent's `tickets`; anything else in
    their state (`open_tickets`, `questions`) is private to them
  * `output_schema` limits what a subgraph returns, so it can't overwrite the parent's other keys
  * both subgraphs report `reviewed_ids` in the same step, so the parent needs a reducer for it
  * streaming with `subgraphs=True` shows events from inside subgraphs, tagged with a namespace
    (`("failures:<task id>",)`). With a checkpointer, `get_state(config, subgraphs=True)` also shows a
    paused subgraph's own state (lesson 4.4 pauses inside the parent).
  * why: separate teams (or later, agents) own separate graphs, tested on their own

Run:  uv run poe lesson 4.2
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.sources import load_sample_tickets


class Review(TypedDict, total=False):  # the parent graph
    tickets: list[dict]
    failure_report: str
    faq_suggestions: str
    reviewed_ids: Annotated[list[str], operator.add]  # written by both subgraphs


class FailureState(TypedDict, total=False):
    tickets: list[dict]  # shared with the parent
    open_tickets: list[dict]  # private to this subgraph
    failure_report: str
    reviewed_ids: list[str]


class FailureOutput(TypedDict):
    failure_report: str
    reviewed_ids: list[str]


class FaqState(TypedDict, total=False):
    tickets: list[dict]
    questions: list[str]
    faq_suggestions: str
    reviewed_ids: list[str]


class FaqOutput(TypedDict):
    faq_suggestions: str
    reviewed_ids: list[str]


def failure_graph(llm: BaseChatModel):
    def find_open(state: FailureState) -> dict:
        return {"open_tickets": [t for t in state["tickets"] if t["status"] == "open"]}

    def write_report(state: FailureState) -> dict:
        lines = "\n".join(f"- {t['summary']} ({t['platform']}, {t['priority']})" for t in state["open_tickets"])
        prompt = f"Group these open Nimbus Notes tickets by likely root cause, in 3 bullets or fewer:\n{lines}"
        return {"failure_report": llm.invoke(prompt).text, "reviewed_ids": [t["id"] for t in state["open_tickets"]]}

    builder = StateGraph(FailureState, output_schema=FailureOutput)
    builder.add_node(find_open)
    builder.add_node(write_report)
    builder.add_edge(START, "find_open")
    builder.add_edge("find_open", "write_report")
    builder.add_edge("write_report", END)
    return builder.compile()


def faq_graph(llm: BaseChatModel):
    def find_questions(state: FaqState) -> dict:
        return {"questions": [t["summary"] for t in state["tickets"] if t["status"] == "solved"]}

    def suggest(state: FaqState) -> dict:
        prompt = "Suggest one FAQ entry title per solved ticket below:\n" + "\n".join(state["questions"])
        solved = [t["id"] for t in state["tickets"] if t["status"] == "solved"]
        return {"faq_suggestions": llm.invoke(prompt).text, "reviewed_ids": solved}

    builder = StateGraph(FaqState, output_schema=FaqOutput)
    builder.add_node(find_questions)
    builder.add_node(suggest)
    builder.add_edge(START, "find_questions")
    builder.add_edge("find_questions", "suggest")
    builder.add_edge("suggest", END)
    return builder.compile()


def build(llm: BaseChatModel):
    builder = StateGraph(Review)
    builder.add_node("failures", failure_graph(llm))  # a compiled graph as a node
    builder.add_node("faq", faq_graph(llm))
    builder.add_edge(START, "failures")
    builder.add_edge(START, "faq")
    builder.add_edge("failures", END)
    builder.add_edge("faq", END)
    return builder.compile()


def run(llm: BaseChatModel) -> dict:
    graph = build(llm)
    tickets = load_sample_tickets()
    header("Streaming with subgraphs=True: (namespace, update)")
    namespaces, state = [], {}
    for namespace, mode, chunk in graph.stream(
        {"tickets": tickets}, stream_mode=["updates", "values"], subgraphs=True
    ):
        if mode == "updates":
            where = namespace[0].split(":")[0] if namespace else "parent"
            namespaces.append(where)
            print(f"  [{where:8}] {', '.join(chunk)}")
        elif not namespace:
            state = chunk
    header("Failure report (subgraph 1)")
    print(state["failure_report"])
    header("FAQ suggestions (subgraph 2)")
    print(state["faq_suggestions"])
    print(f"\nReviewed {len(state['reviewed_ids'])} of {len(tickets)} tickets; parent keys: {sorted(state)}")
    return {**state, "namespaces": namespaces}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm())


if __name__ == "__main__":
    main()
