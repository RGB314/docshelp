"""Connecting to a deployment: the SDK clients, stateless runs, and `RemoteGraph`.

Academy: Introduction to LangGraph M6 L3 "Connecting to a Deployment"; Introduction to LangSmith Deployment
M2 "Connecting", M1 "Use the Tutor".

Concepts:
  * `langgraph_sdk.get_client(url=...)` (async) and `get_sync_client(url=...)` (sync) wrap the server's REST
    API: `.assistants`, `.threads`, `.runs`, `.store`, `.crons`. A deployed server also needs
    `api_key=...` (or `LANGSMITH_API_KEY`); `langgraph dev` doesn't.
  * a *stateless* run: `client.runs.wait(None, "support_agent", input=...)`. No thread, so nothing is
    remembered: right for one-shot jobs (classify this email, summarise this doc).
  * runs take `context=...`, the runtime context of Part 2.2, now sent over HTTP as JSON
  * `RemoteGraph("support_agent", url=...)` is a deployed graph that behaves like a local one: `invoke`,
    `stream`, `get_state`, and it can be a node in a local graph. So teams can deploy agents separately and
    compose them, without changing the calling code.

Use case: a small local "escalation desk" graph decides whether a question needs DocsHelp, and if so, asks
the deployed DocsHelp agent as one of its nodes.

Run:  uv run poe server, then: uv run poe lesson 8.2
"""

import argparse
import asyncio

from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.pregel.remote import RemoteGraph
from langgraph_sdk import get_client

from docshelp.config import ensure_llm_ready
from docshelp.console import header
from docshelp.deployment import connect, last_text, server_url

QUESTION = "How do I export a notebook to PDF?"
ADA = {"name": "Ada", "plan": "team", "account_id": "acct-1001"}


async def ask_async(url: str, question: str) -> str:
    """The async client: the same calls, awaited. Use it inside async apps (FastAPI, bots, notebooks)."""
    client = get_client(url=url)
    result = await client.runs.wait(None, "support_agent", input={"messages": [{"role": "user", "content": question}]})
    return last_text(result)


def escalation_desk(docshelp: RemoteGraph):
    """A local graph with a deployed graph as one of its nodes."""

    def triage(state: MessagesState) -> dict:
        return {}

    def route(state: MessagesState) -> str:
        product_words = ("notebook", "sync", "export", "plan", "price", "account")
        return "docshelp" if any(w in state["messages"][-1].text.lower() for w in product_words) else END

    builder = StateGraph(MessagesState)
    builder.add_node(triage)
    builder.add_node("docshelp", docshelp)  # the remote graph runs on the server; its state comes back here
    builder.add_edge(START, "triage")
    builder.add_conditional_edges("triage", route, ["docshelp", END])
    builder.add_edge("docshelp", END)
    return builder.compile()


def run(client, url: str, question: str = QUESTION) -> dict:
    header("1. A stateless run with the sync client, with runtime context")
    result = client.runs.wait(
        None, "support_agent", input={"messages": [{"role": "user", "content": question}]}, context=ADA
    )
    stateless = last_text(result)
    print(f"Customer: {question}\nDocsHelp: {stateless}")

    header("2. The same with the async client")
    from_async = asyncio.run(ask_async(url, "Which plans include version history?"))
    print(f"DocsHelp: {from_async}")

    header("3. RemoteGraph: a deployed graph used like a local one")
    remote = RemoteGraph("support_agent", url=url)
    thread = client.threads.create()
    config = {"configurable": {"thread_id": thread["thread_id"]}}
    remote.invoke({"messages": [{"role": "user", "content": "My name is Ada."}]}, config)
    answer = remote.invoke({"messages": [{"role": "user", "content": "What's my name?"}]}, config)
    remembered = answer["messages"][-1]["content"]
    print(f"Second turn on the same thread: {remembered}")
    print(f"The server kept {len(remote.get_state(config).values['messages'])} messages in that thread.")

    header("4. RemoteGraph as a node in a local graph")
    desk = escalation_desk(RemoteGraph("support_agent", url=url))
    escalated = desk.invoke({"messages": [{"role": "user", "content": "Sync stopped after the update."}]})
    small_talk = desk.invoke({"messages": [{"role": "user", "content": "Thanks, have a nice day!"}]})
    print(f"Product question -> {len(escalated['messages'])} messages (DocsHelp answered on the server)")
    print(f"Small talk       -> {len(small_talk['messages'])} message (handled locally, no server call)")
    return {
        "stateless": stateless,
        "async": from_async,
        "remembered": remembered,
        "escalated": len(escalated["messages"]),
        "small_talk": len(small_talk["messages"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=server_url(), help="the LangGraph Server, e.g. http://127.0.0.1:2024")
    args = parser.parse_args()
    ensure_llm_ready()
    run(connect(args.url), args.url)


if __name__ == "__main__":
    main()
