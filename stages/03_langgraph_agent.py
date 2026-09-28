"""Stage 3 - LangGraph: a stateful agent with branching, a self-correcting loop, memory and
human-in-the-loop approval.

What LangGraph gives you here (see src/docshelp/graph.py for the graph itself):
  * control flow you define: route by intent, grade retrieved docs, rewrite + retry if needed
  * a checkpointer that saves state after every step, per conversation thread, which gives
      - memory: the follow-up question is resolved using earlier turns
      - interrupts: the run pauses before opening a ticket and resumes after a human decides
      - history: every intermediate state can be inspected (and replayed)

Run:  uv run poe stage3                   (asks you to approve the ticket)
      uv run poe stage3 --auto-approve
      Docker: docker compose run --rm app poe stage3
"""

import argparse
import json
import uuid

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from docshelp.config import PROJECT_ROOT, describe, ensure_llm_ready
from docshelp.graph import build_graph

CONVERSATION = [
    "How long do deleted notes stay in the trash?",
    "And after that, can an admin still get them back?",  # needs memory to resolve "them"
    "Thanks! Different issue: my sync has shown a red slash for two days and resetting the cache "
    "didn't help. Please open a support ticket.",
]


def print_updates(stream) -> None:
    """stream_mode='updates' yields {node_name: state_update} after each node finishes."""
    for chunk in stream:
        for node, update in chunk.items():
            if node == "__interrupt__":
                print("  ⏸  interrupted, waiting for a human")
                continue
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-approve", action="store_true")
    args = parser.parse_args()
    ensure_llm_ready()
    print(f"Using {describe()}")

    graph = build_graph(checkpointer=InMemorySaver())

    mermaid_path = PROJECT_ROOT / "graph.mmd"
    mermaid_path.write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
    print(f"Graph diagram written to {mermaid_path.name} (paste into https://mermaid.live)")

    # Each thread_id is an independent conversation with its own saved state.
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    for turn, text in enumerate(CONVERSATION, 1):
        print(f"\n--- Turn {turn} ---------------------------------------------------------")
        print(f"User: {text}")
        print_updates(graph.stream({"messages": [HumanMessage(text)]}, config, stream_mode="updates"))

        state = graph.get_state(config)
        while state.next:  # the graph is paused at an interrupt
            payload = state.tasks[0].interrupts[0].value
            print(f"\n  {payload['question']}\n  {json.dumps(payload['ticket'], indent=2)}")
            if args.auto_approve:
                decision = "approve"
                print("  (auto-approved)")
            else:
                decision = input("  approve / reject > ").strip().lower() or "approve"
            print_updates(graph.stream(Command(resume={"action": decision}), config, stream_mode="updates"))
            state = graph.get_state(config)

        print(f"\nDocsHelp: {state.values['messages'][-1].text}")

    history = list(graph.get_state_history(config))
    print(
        f"\nThe checkpointer saved {len(history)} snapshots for this thread. "
        "Any of them can be inspected, or resumed from ('time travel')."
    )


if __name__ == "__main__":
    main()
