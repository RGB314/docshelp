"""Capstone: the DocsHelp support graph, with routing, a self-correcting loop, memory and approval.

Academy: LangGraph Essentials L6 "Application" (the email workflow, "Thinking in LangGraph"); Intro to
LangGraph M1 L6–L7 (agent with memory) and M3 (interrupts), brought together.

The graph is `src/docshelp/graph.py::build_graph` (also served to Studio, lesson 3.4). It shows how to
design a real workflow in LangGraph:
  * one node per step, each doing one job: classify → retrieve → grade → (rewrite ↺) → generate,
    or draft_ticket → human_review → open_ticket
  * state holds raw data (intent, documents, ticket), and prompts are built inside the nodes
  * routing with conditional edges (lesson 3.2) and with `Command` from the review node
  * errors handled by kind: `RetryPolicy` for transient failures, a bounded rewrite loop when retrieval
    misses, `interrupt()` for decisions a human must make (lesson 3.13)
  * a checkpointer for multi-turn memory, the paused review, and the history (lesson 3.14)

Run:  uv run poe lesson 3.15             (scripted reviewer approves the ticket)
      uv run poe lesson 3.15 --review    (you approve, or reject)
"""

import argparse
import json
import uuid
from collections.abc import Callable
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from docshelp.config import PROJECT_ROOT, describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.graph import build_graph
from docshelp.ingest import get_retriever

CONVERSATION = [
    "How long do deleted notes stay in the trash?",
    "And after that, can an admin still get them back?",  # needs memory to resolve "them"
    "Thanks! Different issue: my sync has shown a red slash for two days and resetting the cache "
    "didn't help. Please open a support ticket.",
]


def approve(payload: dict) -> dict:
    return {"action": "approve"}


def terminal_review(payload: dict) -> dict:
    return {"action": input("  approve / reject > ").strip().lower() or "approve"}


def show(stream, path: list[str]) -> None:
    """Print each finished node with the most useful part of its update."""
    for chunk in stream:
        for node, update in chunk.items():
            if node == "__interrupt__":
                print("  ⏸  paused for a human")
                continue
            path.append(node)
            detail = ""
            if update and "intent" in update:
                detail = f"intent={update['intent']!r}, question={update['question']!r}"
            elif update and "documents" in update:
                detail = "sources=" + ", ".join(sorted({d["source"] for d in update["documents"]}))
            elif update and "docs_relevant" in update:
                detail = f"relevant={update['docs_relevant']}"
            elif update and "question" in update:
                detail = f"rewritten query={update['question']!r}"
            print(f"  ✓ {node:13} {detail}")


def run(
    llm,
    retriever: BaseRetriever,
    conversation: list[str] | None = None,
    review: Callable[[dict], dict] = approve,
    diagram: Path | None = None,
) -> dict:
    graph = build_graph(llm, retriever, InMemorySaver())
    diagram = diagram or PROJECT_ROOT / "graph.mmd"
    diagram.write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
    print(f"Graph diagram written to {diagram.name} (paste it into https://mermaid.live)")

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}  # one conversation, one thread
    paths = []
    for turn, text in enumerate(conversation or CONVERSATION, 1):
        header(f"Turn {turn}")
        print(f"User: {text}")
        path: list[str] = []
        show(graph.stream({"messages": [HumanMessage(text)]}, config, stream_mode="updates"), path)
        state = graph.get_state(config)
        while state.next:  # paused at the interrupt in human_review
            payload = state.tasks[0].interrupts[0].value
            print(f"\n  {payload['question']}\n  {json.dumps(payload['ticket'], indent=2)}")
            decision = review(payload)
            print(f"  reviewer: {decision['action']}")
            show(graph.stream(Command(resume=decision), config, stream_mode="updates"), path)
            state = graph.get_state(config)
        paths.append(path)
        print(f"\nDocsHelp: {state.values['messages'][-1].text}")

    snapshots = len(list(graph.get_state_history(config)))
    print(f"\nThe checkpointer saved {snapshots} snapshots for this thread (lesson 3.14 travels through them).")
    return {"paths": paths, "snapshots": snapshots}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", action="store_true", help="approve or reject the ticket yourself")
    args = parser.parse_args()
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever(), review=terminal_review if args.review else approve)


if __name__ == "__main__":
    main()
