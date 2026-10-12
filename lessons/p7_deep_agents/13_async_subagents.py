"""Asynchronous subagents and local deployment: background work on an agent server.

Academy: Introduction to Deep Agents M5 "Local Deployment", "Asynchronous Subagents".

Concepts:
  * a `task` subagent (7.11) blocks: the main agent waits for its answer. An `AsyncSubAgent` runs on an
    Agent Protocol server (LangGraph Server, locally or deployed) as a background run. The main agent gets a
    task id at once and carries on, talking to the user, while the subagent works.
  * the spec is `{"name", "description", "graph_id", "url"}`: `graph_id` is a graph the server serves
    (see `langgraph.json`), so the subagent can be a different codebase, scaled and deployed on its own.
  * the main agent gets five tools: `start_async_task`, `check_async_task` (status, and the result when
    done), `update_async_task` (send new instructions to the same task), `cancel_async_task`,
    `list_async_tasks`. Task ids live in its state (`async_tasks`), so they survive summarization.
  * local deployment: `uv run poe server` runs LangGraph Server on your machine with this repo's graphs,
    including `docs_researcher` (the worker below) and `ops_analyst` (the 7.14 capstone). Part 8 covers the
    server in depth.
  * offline twin: without `--url`, `docshelp.deep.LocalAgentServer` stands in for the server, in-process.
    It runs each graph in a background thread and answers the same client calls.

Use case: the lead asks for background research on the Windows sync issue, then checks back later.

Run:  uv run poe lesson 7.13                                   (in-process stand-in server)
      uv run poe server, then: uv run poe lesson 7.13 --url http://127.0.0.1:2024
"""

import argparse
import contextlib

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.memory import InMemorySaver

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.deep import LocalAgentServer, build_docs_researcher, deep_agent, serve_locally
from docshelp.ingest import get_retriever

PROMPT = "You are the Support Ops lead's assistant. Hand long research to the researcher async subagent."
TURNS = [
    "In the background, research why Windows users see a red sync icon after sleep. I'll check back later.",
    "Is the research done? What did it find?",
]


def run(
    llm: BaseChatModel,
    retriever: BaseRetriever | None = None,
    *,
    worker_llm: BaseChatModel | None = None,
    url: str | None = None,
) -> dict:
    researcher = {
        "name": "researcher",
        "description": "Researches a support problem in the docs and changelog, in the background.",
        "graph_id": "docs_researcher",
        "url": url or "local://docshelp",
    }
    agent = deep_agent(llm, subagents=[researcher], system_prompt=PROMPT, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "lead"}}
    if url:
        from langgraph_sdk import get_sync_client

        server = None
        client = get_sync_client(url=url)
        serving = contextlib.nullcontext()
    else:
        server = LocalAgentServer({"docs_researcher": build_docs_researcher(worker_llm or llm, retriever)})
        serving = serve_locally(server)
    with serving:
        result = agent.invoke({"messages": [{"role": "user", "content": TURNS[0]}]}, config)
        print(f"Lead: {TURNS[0]}\nAgent: {result['messages'][-1].text}")
        tasks = result.get("async_tasks", {})
        header(f"Running in the background: {list(tasks)}")
        for task in tasks.values():  # here we simply wait; a real app would let the user keep chatting
            (server.runs_api if server else client.runs).join(task["thread_id"], task["run_id"])
        result = agent.invoke({"messages": [{"role": "user", "content": TURNS[1]}]}, config)
        print(f"Lead: {TURNS[1]}\nAgent: {result['messages'][-1].text}")
    checks = [m.text for m in result["messages"] if isinstance(m, ToolMessage) and m.name == "check_async_task"]
    return {"tasks": result.get("async_tasks", {}), "checks": checks, "answer": result["messages"][-1].text}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="a running LangGraph Server, e.g. http://127.0.0.1:2024 (`uv run poe server`)")
    args = parser.parse_args()
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever(), url=args.url)


if __name__ == "__main__":
    main()
