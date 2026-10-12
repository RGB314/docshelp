"""Threads, runs and streaming over the API, including human approval of a paused run.

Academy: Introduction to LangGraph M6 L2 "Creating a Deployment", L3 "Connecting to a Deployment" (threads,
runs, streaming); Introduction to LangSmith Deployment M1 "Use the Tutor". (Replaces the old stages/05.)

Concepts:
  * threads: `threads.create(metadata=...)`, `threads.search(metadata=...)`, `get_state`, `get_history`,
    `threads.copy` (a fork you can try something else on). The server checkpoints every step.
  * runs:
      - `runs.stream(thread, assistant, input=..., stream_mode=...)`: results as they happen. Modes as in
        Part 3.11: "values", "updates", "messages-tuple" (tokens), "custom", "events", "debug"
      - `runs.wait(...)`: block until done, get the final state
      - `runs.create(...)` + `runs.join(...)`: a *background* run; `runs.list(thread)` and `runs.get` show
        its status (pending, running, success, error, interrupted)
  * human in the loop over HTTP: a run that hits `interrupt()` (Part 3.13) ends with the thread waiting;
    `get_state()["next"]` and `["tasks"][...]["interrupts"]` show what it's waiting for, and a new run with
    `command={"resume": ...}` continues it. Studio and Agent Chat UI use exactly these calls.

Use case: a customer asks a docs question, then asks for a ticket; a support lead approves it via the API.

Run:  uv run poe server, then: uv run poe lesson 8.3
"""

import argparse
import json

from docshelp.config import ensure_llm_ready
from docshelp.console import header
from docshelp.deployment import connect, last_text, server_url

TURNS = [
    "What are the API rate limits on the Team plan?",
    "My webhooks stopped arriving since yesterday. Please open a ticket.",
]


def stream_updates(client, thread_id: str, **kwargs) -> list[str]:
    """Stream a run of the docshelp graph and print each node as it finishes."""
    nodes = []
    for part in client.runs.stream(thread_id, "docshelp", stream_mode="updates", **kwargs):
        if part.event == "updates":
            for node in part.data:
                nodes.append(node)
                print(f"  ✓ {node}" if node != "__interrupt__" else "  ⏸ waiting for a human")
        elif part.event == "error":
            print(f"  ! error: {part.data}")
    return nodes


def run(client) -> dict:
    thread = client.threads.create(metadata={"account_id": "acct-1001", "channel": "web"})
    thread_id = thread["thread_id"]
    print(f"Thread {thread_id} (metadata {thread['metadata']})")

    approvals, answers, nodes = [], [], []
    for text in TURNS:
        print(f"\nCustomer: {text}")
        nodes += stream_updates(client, thread_id, input={"messages": [{"role": "user", "content": text}]})
        state = client.threads.get_state(thread_id)
        if state["next"]:  # paused at the human_review interrupt
            payload = state["tasks"][0]["interrupts"][0]["value"]
            approvals.append(payload["ticket"])
            print(f"  Ticket awaiting approval: {json.dumps(payload['ticket'])}")
            print("  The lead approves it through the API (Studio offers the same as a button)...")
            nodes += stream_updates(client, thread_id, command={"resume": {"action": "approve"}})
            state = client.threads.get_state(thread_id)
        answers.append(last_text(state["values"]))
        print(f"DocsHelp: {answers[-1]}")

    header("Tokens as they're generated (stream_mode='messages-tuple')")
    tokens = 0
    print("DocsHelp: ", end="")
    for part in client.runs.stream(
        None,
        "support_agent",
        input={"messages": [{"role": "user", "content": "In one sentence: what is Nimbus Notes?"}]},
        stream_mode="messages-tuple",
    ):
        if part.event == "messages":
            chunk, _metadata = part.data
            if chunk.get("type") in ("AIMessageChunk", "ai") and isinstance(chunk.get("content"), str):
                print(chunk["content"], end="", flush=True)
                tokens += 1
    print()

    header("A background run: start it, do other work, then join")
    background = client.runs.create(
        thread_id, "docshelp", input={"messages": [{"role": "user", "content": "Is there a Linux app?"}]}
    )
    print(f"Started run {background['run_id']} (status: {background['status']}); the client is free meanwhile.")
    client.runs.join(thread_id, background["run_id"])
    statuses = [r["status"] for r in client.runs.list(thread_id)]
    print(f"Runs on this thread: {statuses}")

    header("History, search and forks")
    history = client.threads.get_history(thread_id, limit=100)
    print(f"{len(history)} checkpoints saved on the thread (Part 3's time travel works on them too).")
    found = client.threads.search(metadata={"account_id": "acct-1001"}, limit=100)
    print(f"Threads for acct-1001: {len(found)}")
    fork = client.threads.copy(thread_id)
    print(f"Forked into {fork['thread_id']}: same history, separate future.")
    return {
        "answers": answers,
        "approvals": approvals,
        "nodes": nodes,
        "tokens": tokens,
        "runs": statuses,
        "checkpoints": len(history),
        "found": len(found),
        "fork": fork["thread_id"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=server_url(), help="the LangGraph Server, e.g. http://127.0.0.1:2024")
    args = parser.parse_args()
    ensure_llm_ready()
    run(connect(args.url))


if __name__ == "__main__":
    main()
