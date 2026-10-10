"""Time travel: browse a thread's history, replay from a past step, and fork it.

Academy: Intro to LangGraph M3 L5 "Time Travel".

Concepts:
  * the checkpointer saves a snapshot after every step. `get_state_history(config)` lists them, newest
    first. Each `StateSnapshot` has `values`, `next` (what would run next) and its own `config` with a
    `checkpoint_id`.
  * replay: `invoke(None, snapshot.config)` re-runs the graph from that snapshot. Steps before it are
    not re-run, and the steps after it run again (the model is called again).
  * fork: `update_state(snapshot.config, {...})` writes a *new* checkpoint branching off that snapshot
    (here: the same question, but for the Enterprise plan), and `invoke(None, fork_config)` continues
    from it. The original branch stays untouched in history.
  * uses: debugging ("what did the model see at step 3?"), reproducing a bad answer, and "what if"
    testing. Studio (lesson 3.4) shows the same history and forking in its UI.

Run:  uv run poe lesson 3.14
"""

import uuid

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.memory import InMemorySaver

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.ingest import get_retriever
from docshelp.react import build_react_agent
from docshelp.tools import make_search_docs, quote_price

QUESTION = "What does the Team plan cost for 12 members, billed monthly?"
WHAT_IF = "What does the Enterprise plan cost for 12 members, billed monthly?"


def run(llm: BaseChatModel, retriever: BaseRetriever) -> dict:
    agent = build_react_agent(llm, [make_search_docs(retriever), quote_price], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    header("1. The original run")
    original = agent.invoke({"messages": [HumanMessage(QUESTION)]}, config)["messages"][-1].text
    print(f"User: {QUESTION}\nDocsHelp: {original}")

    header("2. Browse the history (newest first)")
    history = list(agent.get_state_history(config))
    for snapshot in history:
        last = snapshot.values.get("messages", [])[-1:] or ["(empty)"]
        last_text = getattr(last[0], "text", last[0])
        print(f"  step {snapshot.metadata['step']:>2}  next={snapshot.next!s:14} last={str(last_text)[:60]!r}")
    # The snapshot just after the user's message, before the model was first called:
    start = next(s for s in reversed(history) if s.next == ("assistant",))

    header("3. Replay from that snapshot (the model runs again)")
    replayed = agent.invoke(None, start.config)["messages"][-1].text
    print(f"DocsHelp: {replayed}")

    header("4. Fork: same snapshot, different question")
    question_id = start.values["messages"][0].id  # same id, so add_messages replaces the question
    fork_config = agent.update_state(start.config, {"messages": [HumanMessage(WHAT_IF, id=question_id)]})
    forked = agent.invoke(None, fork_config)["messages"][-1].text
    print(f"User: {WHAT_IF}\nDocsHelp: {forked}")
    print(f"\nThe thread now holds {len(list(agent.get_state_history(config)))} snapshots across its branches.")
    return {"snapshots": len(history), "original": original, "replayed": replayed, "forked": forked}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
