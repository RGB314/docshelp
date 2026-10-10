"""Parallelization: search the docs, the changelog and the status page at once, then answer.

Academy: Intro to LangGraph M4 L1 "Parallelization".

Concepts:
  * fan-out: several edges leaving one node (here START) run their targets in the *same* step, in parallel
  * fan-in: `add_edge([a, b, c], "answer")` waits for *all* of a, b and c, even when one branch has more
    steps than the others (the docs branch searches, then trims)
  * parallel writes to one key need a reducer (lesson 3.6): `context` uses `operator.add`
  * the order of parallel results isn't guaranteed, so each item is labelled with its source
  * if one branch fails, the whole step fails and nothing from it is saved; retries (lesson 3.15) apply
    per node

Run:  uv run poe lesson 4.1
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, START, StateGraph

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.ingest import get_retriever
from docshelp.sources import search_changelog, status_summary

QUESTION = "Since the update my Windows tray icon shows a red slash after sleep. Is sync broken?"


class State(TypedDict, total=False):
    question: str
    docs: list
    context: Annotated[list[str], operator.add]  # every branch appends; nothing is overwritten
    answer: str


def build(llm: BaseChatModel, retriever: BaseRetriever):
    def search_docs(state: State) -> dict:
        return {"docs": retriever.invoke(state["question"])}

    def trim_docs(state: State) -> dict:  # a second step on the docs branch: the branches are uneven
        top = state["docs"][:2]
        return {"context": [f"[docs] {d.metadata.get('source', '?')}: {d.page_content[:300]}" for d in top]}

    def search_changelog_node(state: State) -> dict:
        return {"context": [f"[changelog] {entry}" for entry in search_changelog(state["question"])]}

    def check_status(state: State) -> dict:
        return {"context": [f"[status] {status_summary()}"]}

    def answer(state: State) -> dict:
        context = "\n".join(sorted(state["context"]))  # sorted: parallel order is not guaranteed
        prompt = (
            "You are DocsHelp, the Nimbus Notes support assistant. Answer using the sources below, and say "
            f"which source each fact comes from.\n\n{context}\n\nQuestion: {state['question']}"
        )
        return {"answer": llm.invoke(prompt).text}

    builder = StateGraph(State)
    builder.add_node("search_docs", search_docs)
    builder.add_node("trim_docs", trim_docs)
    builder.add_node("search_changelog", search_changelog_node)
    builder.add_node("check_status", check_status)
    builder.add_node("answer", answer)
    for branch in ("search_docs", "search_changelog", "check_status"):  # fan-out
        builder.add_edge(START, branch)
    builder.add_edge("search_docs", "trim_docs")
    builder.add_edge(["trim_docs", "search_changelog", "check_status"], "answer")  # fan-in: wait for all
    builder.add_edge("answer", END)
    return builder.compile()


def run(llm: BaseChatModel, retriever: BaseRetriever) -> dict:
    graph = build(llm, retriever)
    header("The graph")
    print(graph.get_graph().draw_mermaid())
    header(f"User: {QUESTION}")
    by_step: dict[int, set[str]] = {}
    state: dict = {}
    # "debug" events carry the step number of each task, which shows which nodes ran together.
    for mode, chunk in graph.stream({"question": QUESTION}, stream_mode=["debug", "values"]):
        if mode == "debug" and chunk["type"] == "task":
            by_step.setdefault(chunk["step"], set()).add(chunk["payload"]["name"])
        elif mode == "values":
            state = chunk  # the full state after each step; the last one is the result
    steps = [by_step[n] for n in sorted(by_step)]
    for n, nodes in enumerate(steps, 1):
        print(f"  step {n}: {', '.join(sorted(nodes))}{'   <- in parallel' if len(nodes) > 1 else ''}")
    header("Context gathered by the three branches")
    for line in sorted(state["context"]):
        print(f"  {line[:150]}")
    print(f"\nDocsHelp: {state['answer']}")
    return {"steps": steps, "context": state["context"], "answer": state["answer"]}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
