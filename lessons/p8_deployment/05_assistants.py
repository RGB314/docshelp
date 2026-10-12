"""Assistants: one deployed graph, several configurations, each versioned.

Academy: Introduction to LangGraph M6 L5 "Assistants"; Introduction to LangSmith Deployment M2 "Assistants".

Concepts:
  * an *assistant* is a graph plus saved settings: `context` (the runtime context, Part 2.2), `config`,
    `metadata`, a name and a description. Clients run an assistant by id; the server merges its settings
    into every run (the run's own `context` adds to them).
  * so product variants need no new code or deploy: here `support_agent` becomes a "concise" assistant for
    chat and a "detailed" one for email, by `AssistantContext.style` (and `model`: "fast" picks the smaller
    model, as in lesson 2.8). Non-developers can do the same from Studio's assistant editor.
  * versioning: `assistants.update(...)` saves a new version and makes it current; `get_versions` lists
    them; `set_latest(id, version)` rolls back (or forward) instantly, with no redeploy.
  * `assistants.search(metadata=..., graph_id=...)` finds them; `assistants.delete` removes one.
  * `get_schemas` shows what an assistant accepts, so UIs can render a settings form.

Use case: the support team wants short chat answers and step-by-step email answers from the same agent.

Run:  uv run poe server, then: uv run poe lesson 8.5
"""

import argparse

from docshelp.config import ensure_llm_ready
from docshelp.console import header
from docshelp.deployment import connect, last_text, server_url

QUESTION = "How do I share a notebook with a colleague?"
CUSTOMER = {"name": "Ada", "plan": "team", "account_id": "acct-1001"}


def ask(client, assistant_id: str) -> str:
    result = client.runs.wait(
        None, assistant_id, input={"messages": [{"role": "user", "content": QUESTION}]}, context=CUSTOMER
    )
    return last_text(result)


def run(client) -> dict:
    header("Two assistants from one graph")
    concise = client.assistants.create(
        "support_agent",
        name="docshelp-chat",
        description="Short answers for the in-app chat.",
        context={"style": "concise", "model": "fast"},
        metadata={"channel": "chat", "team": "support"},
    )
    detailed = client.assistants.create(
        "support_agent",
        name="docshelp-email",
        description="Step-by-step answers for email replies.",
        context={"style": "detailed"},
        metadata={"channel": "email", "team": "support"},
    )
    answers = {}
    for assistant in (concise, detailed):
        answers[assistant["name"]] = ask(client, assistant["assistant_id"])
        print(f"\n[{assistant['name']}] context={assistant['context']}\n{answers[assistant['name']]}")

    header("Versions: change an assistant, then roll back")
    updated = client.assistants.update(
        detailed["assistant_id"], context={"style": "detailed", "model": "fast"}, metadata={"channel": "email"}
    )
    versions = {v["version"]: v["context"] for v in client.assistants.get_versions(detailed["assistant_id"])}
    print(f"docshelp-email is now version {updated['version']}. Every version is kept:")
    for version, context in sorted(versions.items()):
        print(f"  v{version}: context {context}")
    client.assistants.set_latest(detailed["assistant_id"], 1)
    rolled_back = client.assistants.get(detailed["assistant_id"])  # what runs use from now on
    print(f"Rolled back: the assistant is at version {rolled_back['version']} again.")
    if rolled_back["context"] != versions[1]:
        # Seen with `langgraph dev`'s in-memory runtime (0.35): set_latest restores config, metadata and name,
        # but not context. Check `assistants.get` after a rollback if your settings live in context.
        print(f"  Note: this server still reports context {rolled_back['context']}; version 1 has {versions[1]}.")

    header("Finding assistants")
    found = client.assistants.search(metadata={"team": "support"}, graph_id="support_agent")
    print(f"Support team assistants on support_agent: {sorted(a['name'] for a in found)}")
    context_schema = client.assistants.get_schemas(concise["assistant_id"])["context_schema"]
    print(f"Settings a UI could offer: {list(context_schema.get('properties', {}))}")

    for assistant in (concise, detailed):  # tidy up, so the lesson can run again from scratch
        client.assistants.delete(assistant["assistant_id"])
    return {
        "answers": answers,
        "versions": versions,
        "rolled_back_to": rolled_back["version"],
        "found": sorted(a["name"] for a in found),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=server_url(), help="the LangGraph Server, e.g. http://127.0.0.1:2024")
    args = parser.parse_args()
    ensure_llm_ready()
    run(connect(args.url))


if __name__ == "__main__":
    main()
