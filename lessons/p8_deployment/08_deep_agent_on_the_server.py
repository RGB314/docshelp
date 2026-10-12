"""A deep agent on the server: approvals over the API, and `AGENTS.md` memory in the server's Store.

Academy: Introduction to LangSmith Deployment M4 "Memory in Deep Agents", M6 "Deploy the Tutor Agent".

Concepts:
  * the Part 7 capstone, `build_ops_analyst(on_server=True)`, is served as `ops_analyst`. It is compiled
    without a store or checkpointer, so its `StoreBackend` (/memories/) uses the server's Store and its
    threads use the server's checkpointer. Nothing else changes between the lesson and the deployment.
  * deep-agent memory is just Store items: `/memories/AGENTS.md` is the item `/AGENTS.md` in the namespace
    `("support-ops", "memories")`, value `{"content": ..., "encoding": "utf-8"}`. So you can seed it, read
    it, review it or roll it back with `client.store`, without touching the agent.
  * `MemoryMiddleware` reads AGENTS.md when a thread starts: what the agent learns in one thread applies in
    every later thread, for every user of the deployment (that's why it's a team-wide namespace).
  * HITL over the API: the report write pauses the run (`FilesystemPermission(mode="interrupt")`, 7.4); the
    interrupt value lists `action_requests`; a run with `command={"resume": {"decisions": [...]}}` continues.
  * the same server also runs the async subagents of lesson 7.13 (`uv run poe lesson 7.13 --url ...`).

Use case: the lead asks the deployed Ops analyst for a report, approves it, and gives lasting feedback.

Run:  uv run poe server, then: uv run poe lesson 8.8
"""

import argparse

from docshelp.config import ensure_llm_ready
from docshelp.console import header
from docshelp.deep import DEFAULT_MEMORY, MEMORY_PATH, team_memory_namespace
from docshelp.deployment import connect, last_text, server_url

NAMESPACE = list(team_memory_namespace(None))
KEY = MEMORY_PATH.removeprefix("/memories")  # "/AGENTS.md"
REQUEST = "Write a short incident report on the Windows sync tickets to /reports/windows-sync.md."
FEEDBACK = "Good. From now on, start every report with a one-line TL;DR. Remember that for next time."
CHECK = "What conventions does the team want in reports?"
MAX_APPROVALS = 5


def memory(client) -> str:
    item = client.store.get_item(NAMESPACE, KEY)
    return item["value"]["content"] if item else ""


def turn(client, thread_id: str, text: str, approvals: list) -> str:
    """One message to the analyst; approve every action it pauses for, as the lead would in Studio."""
    client.runs.wait(thread_id, "ops_analyst", input={"messages": [{"role": "user", "content": text}]})
    state = client.threads.get_state(thread_id)
    while state["next"] and len(approvals) < MAX_APPROVALS:
        request = state["tasks"][0]["interrupts"][0]["value"]
        actions = request["action_requests"]
        for action in actions:
            approvals.append(action["args"].get("file_path", action["name"]))
            print(f"  ⏸ approve {action['name']} {action['args'].get('file_path', '')}")
        decisions = [{"type": "approve"} for _ in actions]
        client.runs.wait(thread_id, "ops_analyst", command={"resume": {"decisions": decisions}})
        state = client.threads.get_state(thread_id)
    return last_text(state["values"])


def run(client) -> dict:
    header("1. Seed the team memory through the Store API (once)")
    if not memory(client):
        client.store.put_item(NAMESPACE, KEY, {"content": DEFAULT_MEMORY, "encoding": "utf-8"})
    before = memory(client)
    print(before)

    header("2. A report, approved over the API, then feedback")
    approvals = []
    thread_id = client.threads.create(metadata={"team": "support-ops"})["thread_id"]
    for text in (REQUEST, FEEDBACK):
        print(f"\nLead: {text}")
        print(f"Analyst: {turn(client, thread_id, text, approvals)}")

    header("3. The memory after the conversation, read with the SDK")
    after = memory(client)
    print(after)
    print("(changed)" if after != before else "(unchanged: the model didn't save the feedback this time)")

    header("4. A new thread starts from the updated memory")
    new_thread = client.threads.create(metadata={"team": "support-ops"})["thread_id"]
    print(f"Lead: {CHECK}")
    print(f"Analyst: {turn(client, new_thread, CHECK, approvals)}")
    return {"before": before, "after": after, "approvals": approvals}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=server_url(), help="the LangGraph Server, e.g. http://127.0.0.1:2024")
    args = parser.parse_args()
    ensure_llm_ready()
    run(connect(args.url))


if __name__ == "__main__":
    main()
