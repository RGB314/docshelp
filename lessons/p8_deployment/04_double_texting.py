"""Double texting: what the server does when a new message arrives while the agent is still working.

Academy: Introduction to LangGraph M6 L4 "Double Texting".

Concepts:
  * people send a second message before the first answer arrives ("sync is broken" ... "on Windows btw").
    A thread runs one thing at a time, so `multitask_strategy` on the new run decides what happens:
      - "reject":    refuse the new run (HTTP 409). The UI should disable input while a run is in flight.
      - "enqueue":   queue it; it runs after the current one, which finishes normally. (The default.)
      - "interrupt": stop the current run where it is, keep its progress, start the new one. The thread
                     keeps both messages; the first run ends as "interrupted".
      - "rollback":  stop the current run *and undo it* (its run and checkpoints are deleted), then start
                     the new one as if the first message had never been sent.
  * the same strategies apply to `runs.stream` and `runs.wait`; crons and webhooks reuse the queue.

Use case: a customer sends "Sync is broken." and then, while DocsHelp is still answering, "On Windows."

Run:  uv run poe server, then: uv run poe lesson 8.4
"""

import argparse
import time

from langgraph_sdk.errors import ConflictError

from docshelp.config import ensure_llm_ready
from docshelp.console import header
from docshelp.deployment import connect, server_url

FIRST = "Sync is broken."
SECOND = "On Windows, if that matters."
STRATEGIES = ("reject", "enqueue", "interrupt", "rollback")


def message(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


def wait_until_running(client, thread_id: str, run_id: str, timeout: float = 30) -> str:
    """Poll the first run until a worker has picked it up, so the second message really is a double text."""
    deadline = time.monotonic() + timeout
    status = client.runs.get(thread_id, run_id)["status"]
    while status == "pending" and time.monotonic() < deadline:
        time.sleep(0.1)
        status = client.runs.get(thread_id, run_id)["status"]
    return status


def double_text(client, strategy: str) -> dict:
    thread_id = client.threads.create()["thread_id"]
    first = client.runs.create(thread_id, "support_agent", input=message(FIRST))
    first_status = wait_until_running(client, thread_id, first["run_id"])
    try:
        second = client.runs.create(thread_id, "support_agent", input=message(SECOND), multitask_strategy=strategy)
    except ConflictError:
        client.runs.join(thread_id, first["run_id"])
        second = None
    if second:
        client.runs.join(thread_id, second["run_id"])
    runs = {r["run_id"]: r["status"] for r in client.runs.list(thread_id)}
    values = client.threads.get_state(thread_id)["values"]
    customer = [m["content"] for m in values.get("messages", []) if m["type"] == "human"]
    outcome = {
        "first_was": first_status,
        "second": "rejected" if second is None else runs.get(second["run_id"]),
        "first": runs.get(first["run_id"], "deleted"),
        "customer_messages": customer,
    }
    print(
        f"  {strategy:9} first run: {outcome['first']:11} second run: {outcome['second']:9} "
        f"the thread holds: {customer}"
    )
    return outcome


def run(client) -> dict:
    header(f"'{FIRST}' then, while the agent works, '{SECOND}'")
    results = {strategy: double_text(client, strategy) for strategy in STRATEGIES}
    print(
        "\nreject keeps only the first message; enqueue answers both, in order; interrupt keeps both but cuts the "
        "first answer short; rollback forgets the first message entirely."
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=server_url(), help="the LangGraph Server, e.g. http://127.0.0.1:2024")
    args = parser.parse_args()
    ensure_llm_ready()
    run(connect(args.url))


if __name__ == "__main__":
    main()
