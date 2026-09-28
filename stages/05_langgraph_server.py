"""Stage 5 - LangGraph Server + SDK: the same graph, served as an API.

What LangGraph Server gives you here:
  * `langgraph dev` turns the graph in langgraph.json into an HTTP API with assistants, threads,
    runs, streaming, persistence and human-in-the-loop, with no web code on your side
  * LangGraph Studio (opened by `langgraph dev`) to visualise the graph, chat with it, inspect
    state at every step, edit state and re-run from any checkpoint
  * `langgraph_sdk`: a client for that API (Python and JS). The same code works against a
    LangSmith Deployment (LangGraph Platform) in the cloud; only the URL changes.

Run in two terminals:
  1)  uv run poe server           (Docker: docker compose up -d server)
  2)  uv run poe stage5           (Docker: docker compose run --rm app poe stage5)
"""

import asyncio
import json
import os

from langgraph_sdk import get_client

from docshelp.console import ensure_utf8

SERVER_URL = os.getenv("LANGGRAPH_URL", "http://127.0.0.1:2024")
ASSISTANT = "docshelp"  # graph name from langgraph.json


async def stream_run(client, thread_id: str, **kwargs) -> None:
    async for part in client.runs.stream(thread_id, ASSISTANT, stream_mode="updates", **kwargs):
        if part.event == "updates":
            for node in part.data:
                print(f"  ✓ {node}" if node != "__interrupt__" else "  ⏸  interrupted")
        elif part.event == "error":
            print(f"  ! error: {part.data}")


async def main() -> None:
    client = get_client(url=SERVER_URL)
    try:
        assistants = await client.assistants.search(graph_id=ASSISTANT)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Can't reach LangGraph Server at {SERVER_URL}. Start it first with `uv run poe server` (or `docker compose up -d server`).\n{exc}")
    print(f"Connected. Assistant: {assistants[0]['assistant_id']} (graph '{ASSISTANT}')")

    # A thread is a persisted conversation; the server stores its checkpoints.
    thread = await client.threads.create()
    thread_id = thread["thread_id"]
    print(f"Created thread {thread_id}")

    for text in [
        "What are the API rate limits on the Team plan?",
        "My webhooks stopped arriving since yesterday. Please open a ticket.",
    ]:
        print(f"\nUser: {text}")
        await stream_run(client, thread_id, input={"messages": [{"role": "user", "content": text}]})

        state = await client.threads.get_state(thread_id)
        if state["next"]:  # paused at the human_review interrupt
            payload = state["tasks"][0]["interrupts"][0]["value"]
            print(f"  Ticket awaiting approval: {json.dumps(payload['ticket'])}")
            print("  Approving via the API (in Studio you could approve it from the UI instead)...")
            await stream_run(client, thread_id, command={"resume": {"action": "approve"}})
            state = await client.threads.get_state(thread_id)

        print(f"DocsHelp: {state['values']['messages'][-1]['content']}")

    print(
        "\nOpen LangGraph Studio (the URL `langgraph dev` printed), pick this thread and step "
        "through each node's state. With LangSmith configured, the runs appear as traces too."
    )


if __name__ == "__main__":
    ensure_utf8()
    asyncio.run(main())
