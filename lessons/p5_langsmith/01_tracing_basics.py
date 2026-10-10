"""Tracing basics: see every step your agent took, with timings and token counts.

Academy: Intro to Agent Observability & Evaluations M1 L1 "Tracing Basics"; Building Reliable Agents
M1 L1 "Observability", L2 "Tracing with LangSmith"; LangSmith Essentials M1 "What is Tracing?".

Concepts:
  * a *trace* is the record of one request: a tree of *runs* (the agent, each graph node, each model call,
    each tool call), each with inputs, outputs, latency, token usage and errors
  * turning it on is configuration, not code: `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` (free
    Developer account), optionally `LANGSMITH_PROJECT`. Every LangChain and LangGraph call is then traced
    automatically, in every lesson.
  * `src/docshelp/observability.py::traced` wraps one request as a root run. With LangSmith configured
    it goes to your project; without it, the same SDK builds the same tree locally (the local twin) and
    writes it to `data/traces.jsonl`.
  * why: agents fail in the middle (a wrong tool argument, an empty search). A trace shows where.

Run:  uv run poe lesson 5.1
"""

import uuid

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.memory import InMemorySaver

from docshelp.agents import Customer, build_support_agent
from docshelp.config import describe, ensure_llm_ready, get_llm, langsmith_enabled
from docshelp.console import header
from docshelp.ingest import get_retriever
from docshelp.observability import TRACES_PATH, print_tree, total_tokens, traced

QUESTION = "What would 12 seats on the Team plan cost per year, billed annually?"


def run(llm: BaseChatModel, retriever: BaseRetriever, traces_path=None) -> dict:
    agent = build_support_agent(llm, retriever, InMemorySaver(), structured=False)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    customer = Customer(name="Ada", plan="team", account_id="acct-1001")
    local = f"memory only, saved to {(traces_path or TRACES_PATH).name}"
    where = "your LangSmith project" if langsmith_enabled() else local
    header(f"Tracing one request ({where})")
    print(f"User: {QUESTION}")
    with traced("support_request", {"question": QUESTION}, metadata={"plan": customer.plan}, path=traces_path) as run:
        result = agent.invoke({"messages": [{"role": "user", "content": QUESTION}]}, config, context=customer)
        answer = result["messages"][-1].text
        run.end(outputs={"answer": answer})
    print(f"DocsHelp: {answer}")

    header("The trace: one root run and its children")
    print_tree(run)
    print(f"\nTotal tokens: {total_tokens(run)}")
    if langsmith_enabled():
        print("Open LangSmith → Tracing projects → your project to explore the same tree in the UI.")
    else:
        print("Set LANGSMITH_TRACING=true and LANGSMITH_API_KEY in .env to send traces to LangSmith (free).")
    return {"run": run, "answer": answer}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
