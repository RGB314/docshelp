"""Deploying an ambient agent: ingest events as background runs, schedule it with a cron, review in an inbox.

Academy: Project: Ambient Agents with LangGraph M6 "Deployment".

Concepts:
  * an ambient agent has no chat box: events start it. Deployed, each event (an email) becomes a
    background run of `inbox_agent` (Part 6) on its own thread. The thread id is derived from the email id
    (`uuid5`), so ingesting the same email again finds its thread and does nothing: ingestion is idempotent.
  * `inbox_ingest` is the graph that does the ingesting. It runs *on the server* and calls the server back
    with `get_client()` (no URL = loopback). Run it once by hand, or on a schedule:
  * crons: `client.crons.create("inbox_ingest", schedule="*/15 * * * *", input=...)` runs it every 15
    minutes, in UTC (`timezone=` to change). `crons.search`, `crons.update` (e.g. `enabled=False`) and
    `crons.delete` manage them. `create_for_thread` runs on one thread instead of a new one each time.
  * the human side is asynchronous too: runs that need a person stop with an interrupt in Agent Inbox's
    format (Part 6.3). An inbox UI lists them with `threads.search(status="interrupted")` and answers with
    `command={"resume": [response]}`. Agent Inbox (https://dev.agentinbox.ai, open source) does exactly
    that: add an inbox with the deployment URL and graph id `inbox_agent`.

Use case: new support emails arrive; DocsHelp drafts answers in the background; the lead reviews the
drafts that need approval, whenever they get to it.

Run:  uv run poe server, then: uv run poe lesson 8.12
"""

import argparse
import time

from docshelp.config import ensure_llm_ready
from docshelp.console import header
from docshelp.deployment import connect, server_url

EMAILS = 3  # how many inbox emails to ingest (all 8 take a while on a small local model)
SCHEDULE = "*/15 * * * *"


def wait_for_threads(client, thread_ids: list[str], timeout: float = 1200) -> dict[str, str]:
    """Background runs finish on their own; here we poll each thread until it's idle, interrupted or failed."""
    deadline = time.monotonic() + timeout
    statuses = {}
    while time.monotonic() < deadline:
        statuses = {t: client.threads.get(t)["status"] for t in thread_ids}
        if all(s != "busy" for s in statuses.values()):
            break
        time.sleep(1)
    return statuses


def run(client, emails: int = EMAILS) -> dict:
    header("1. Ingest: one background run per new email")
    result = client.runs.wait(None, "inbox_ingest", input={"limit": emails})
    started = result["started"]
    print(f"Started inbox_agent on: {started or '(nothing new)'}")
    again = client.runs.wait(None, "inbox_ingest", input={"limit": emails})["started"]
    print(f"Ingesting again starts: {again or 'nothing (same emails, same threads)'}")
    threads = client.threads.search(metadata={"source": "inbox"}, limit=100)
    statuses = wait_for_threads(client, [t["thread_id"] for t in threads])
    print(f"Thread statuses: {sorted(statuses.values())}")

    header("2. Schedule it: a cron")
    cron = client.crons.create("inbox_ingest", schedule=SCHEDULE, input={"limit": emails})
    print(f"Cron {cron['cron_id']} runs '{cron['schedule']}' (UTC); next run at {cron['next_run_date']}")
    crons = client.crons.search(assistant_id=cron["assistant_id"])
    client.crons.delete(cron["cron_id"])  # this is a demo: don't leave it polling
    print(f"Crons for inbox_ingest: {len(crons)}; deleted the demo cron again.")

    header("3. The human inbox: runs waiting for a person")
    waiting = client.threads.search(metadata={"source": "inbox"}, status="interrupted", limit=100)
    for thread in waiting:
        request = thread["interrupts"] and next(iter(thread["interrupts"].values()))[0]["value"][0]
        print(f"  {thread['metadata']['email_id']}  {thread['metadata']['subject']!r}: {request['action_request']['action']}")
    resumed = None
    if waiting:  # accept the first draft, as the lead would by clicking "Accept" in Agent Inbox
        resumed = waiting[0]["thread_id"]
        client.runs.wait(resumed, "inbox_agent", command={"resume": [{"type": "accept", "args": None}]})
        after = client.threads.get(resumed)["status"]
        print(f"Accepted the first one; that thread is now {after}. The rest wait in the inbox.")
    return {
        "started": started,
        "again": again,
        "statuses": statuses,
        "cron_schedule": cron["schedule"],
        "waiting": [t["metadata"]["email_id"] for t in waiting],
        "resumed": resumed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=server_url(), help="the LangGraph Server, e.g. http://127.0.0.1:2024")
    parser.add_argument("--emails", type=int, default=EMAILS, help="how many inbox emails to ingest")
    args = parser.parse_args()
    ensure_llm_ready()
    run(connect(args.url), args.emails)


if __name__ == "__main__":
    main()
