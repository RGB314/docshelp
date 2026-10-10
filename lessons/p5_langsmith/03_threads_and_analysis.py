"""Conversational threads, metadata and tags, and analysing an agent from its traces.

Academy: Intro to Agent Observability & Evaluations M1 L4 "Conversational Threads"; Building Reliable
Agents M1 L3 "Analyzing your Agent".

Concepts:
  * each turn of a chat is its own trace. LangSmith groups traces into a *thread* when they share a
    `thread_id` (or `session_id` / `conversation_id`) in their metadata, and the Threads view then shows
    the whole conversation with its total tokens and latency.
  * metadata (key/value: plan, app version, customer) and tags (labels: "beta", "email") make traces
    filterable. Set them per call (`config={"metadata": ...}`) or per block (`tracing_context`).
  * analysing an agent: find the slowest steps, the most expensive calls, and the errors. In LangSmith
    use the filter bar, or `Client().list_runs(project_name=..., filter='and(eq(run_type, "llm"),
    gt(latency, 5))')`. The local twin reads the same fields from `data/traces.jsonl`.

Run:  uv run poe lesson 5.3
"""

import uuid
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.memory import InMemorySaver

from docshelp.agents import Customer, build_support_agent
from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.ingest import get_retriever
from docshelp.observability import TRACES_PATH, load_rows, summarize_threads, traced

CONVERSATIONS = [  # (customer, their turns)
    (Customer(name="Ada", plan="team", account_id="acct-1001"), [
        "How do I export a notebook as PDF?",
        "And can I include the attachments?",
    ]),
    (Customer(name="Sam", plan="free"), [
        "Does the free plan include offline mode?",
        "What about on my phone?",
    ]),
]


def run(llm: BaseChatModel, retriever: BaseRetriever, traces_path: Path | None = None) -> dict:
    traces_path = traces_path or TRACES_PATH
    agent = build_support_agent(llm, retriever, InMemorySaver(), structured=False)
    session_threads = []
    for customer, turns in CONVERSATIONS:
        thread = str(uuid.uuid4())
        session_threads.append(thread)
        header(f"{customer.name}'s conversation (thread {thread[:8]})")
        for text in turns:
            metadata = {"thread_id": thread, "plan": customer.plan, "app_version": "4.12"}
            inputs = {"question": text}
            with traced("support_turn", inputs, metadata=metadata, tags=[customer.plan], path=traces_path) as run:
                config = {"configurable": {"thread_id": thread}}
                result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config, context=customer)
                run.end(outputs={"answer": result["messages"][-1].text})
            print(f"User: {text}\nDocsHelp: {result['messages'][-1].text}")

    all_rows = load_rows(traces_path)  # may hold older traces too: keep only this session's
    roots = [r for r in all_rows if r["parent_id"] is None]
    ours = {r["trace_id"] for r in roots if r["metadata"].get("thread_id") in session_threads}
    rows = [r for r in all_rows if r["trace_id"] in ours]
    threads = summarize_threads(rows)
    header("Threads (one row per conversation)")
    for thread, summary in threads.items():
        print(f"  {thread[:8]}  turns={summary['turns']}  tokens={summary['tokens']}  "
              f"latency={summary['latency_ms']:.0f} ms  errors={summary['errors']}")

    header("Analysing the agent: the slowest model calls")
    llm_rows = sorted((r for r in rows if r["run_type"] == "llm"), key=lambda r: -(r["latency_ms"] or 0))
    for r in llm_rows[:3]:
        print(f"  {r['latency_ms']:>8.1f} ms  {r['tokens'] or '?':>5} tokens  {r['name']}")
    return {"threads": threads, "llm_calls": len(llm_rows)}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
