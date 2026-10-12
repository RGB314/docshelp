"""Memory on the server: the deployment's Store, used by the agent and by your own code.

Academy: Introduction to LangSmith Deployment M4 "Memory In LangGraph", "Accessing Storage".

Concepts:
  * short-term memory (a thread's messages) and long-term memory (the Store, Part 4) both come with the
    server: graphs compiled with no checkpointer and no store get the server's. With `langgraph dev` they
    live in memory (and a local `.langgraph_api/` folder); deployed, they're in Postgres.
  * the Store in `langgraph.json`: `"store": {"index": {"embed": ..., "dims": 384, "fields": ["$"]}}` makes
    items searchable by meaning. `embed` is `docshelp.deployment.embed`: the local fastembed model, so
    semantic memory stays free. (`"ttl"` can expire items; `put_item(..., ttl=minutes)` per item.)
  * accessing storage from outside the agent, through the SDK: `client.store.put_item`, `get_item`,
    `search_items(namespace_prefix, query=..., filter=...)`, `list_namespaces`, `delete_item`. A CRM sync
    job, an admin page or a migration script can read and write the same memories the agent uses.
  * Part 4's memory agent (`memory_agent`) runs unchanged: it remembers across threads because each run's
    `context` names the customer, and its memories are namespaced `(kind, account_id)`.

Use case: the CRM pushes a past issue for Ada; Ada chats twice, in two threads; support staff search her
memories by meaning.

Run:  uv run poe server, then: uv run poe lesson 8.7
"""

import argparse

from docshelp.config import ensure_llm_ready
from docshelp.console import header
from docshelp.deployment import connect, last_text, server_url

ADA = {"name": "Ada", "plan": "team", "account_id": "acct-1001"}
CRM_ISSUE = {
    "summary": "Sync stuck after the laptop wakes from sleep",
    "platform": "Windows",
    "status": "solved",
    "resolution": "Updated to Nimbus Notes 4.2",
}
TURNS = [
    "Hi, I'm Ada, a workspace admin. I use Windows at work and an iPad at home. Please remember that.",
    "Which devices do I use, and have I had sync problems before?",
]


def run(client) -> dict:
    header("1. Accessing storage from outside: a CRM sync job writes a memory")
    client.store.put_item(["issues", ADA["account_id"]], "crm-1", CRM_ISSUE)
    print(f"put_item(['issues', '{ADA['account_id']}'], 'crm-1', ...)")

    header("2. The memory agent, two conversations in two threads")
    answers = []
    for text in TURNS:
        thread = client.threads.create(metadata={"account_id": ADA["account_id"]})
        result = client.runs.wait(
            thread["thread_id"], "memory_agent", input={"messages": [{"role": "user", "content": text}]}, context=ADA
        )
        answers.append(last_text(result))
        print(f"\n[new thread] Ada: {text}\nDocsHelp: {answers[-1]}")

    header("3. What the server's Store now holds about Ada")
    namespaces = client.store.list_namespaces(suffix=[ADA["account_id"]])["namespaces"]
    print(f"Namespaces: {namespaces}")
    profile = client.store.get_item(["profile", ADA["account_id"]], "profile")
    print(f"Profile: {profile['value'] if profile else '(the model saved none this time)'}")

    header("4. Semantic search over memories (the store's index)")
    hits = client.store.search_items(["issues"], query="problems after waking the computer", limit=3)["items"]
    for item in hits:
        print(f"  {item['score']:.2f}  {item['namespace']} {item['key']}: {item['value']['summary']}")
    solved = client.store.search_items(["issues", ADA["account_id"]], filter={"status": "solved"})["items"]
    print(f"Filter status=solved: {[i['key'] for i in solved]}")
    return {
        "answers": answers,
        "namespaces": namespaces,
        "profile": profile["value"] if profile else None,
        "hits": [i["key"] for i in hits],
        "solved": [i["key"] for i in solved],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=server_url(), help="the LangGraph Server, e.g. http://127.0.0.1:2024")
    args = parser.parse_args()
    ensure_llm_ready()
    run(connect(args.url))


if __name__ == "__main__":
    main()
