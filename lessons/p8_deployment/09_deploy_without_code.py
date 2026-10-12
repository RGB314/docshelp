"""Deploying without coding: Markdown agent templates become assistants of one deployed graph factory.

Academy: Introduction to LangSmith Deployment M6 "Deploying Without Coding Using deepagents-cli".

Concepts:
  * a *graph factory*: `langgraph.json` may point at a function instead of a graph. The server calls it for
    each run with the run's `config` (and/or a `ServerRuntime`: the user, the store, the access context) and
    uses the graph it returns. See `agent_from_template` in `docshelp/server.py`.
  * here the factory builds a deep agent from a Markdown template in agents/ (lesson 7.15's Fleet twin),
    named by the assistant's `config.configurable.template`. Deploying a new agent is then: write a
    Markdown file, create an assistant. No Python, no redeploy (on a real deployment, a git push adds the
    file; `langgraph dev` reloads it at once).
  * the Academy shows the same idea with `deepagents deploy`, which turned an `AGENTS.md` plus `skills/`
    folder into a deployment. That CLI is now deprecated in favour of Managed Deep Agents (`mda`, private
    preview; its project layout is mapped onto this repo in lesson 7.15). The factory below is the free,
    self-hosted way to get the same result.
  * factories cost a graph build per run, so keep them cheap: file reads go to a worker thread
    (`asyncio.to_thread`), because blocking calls on the server's event loop slow every user down
    (`langgraph dev` reports them as `BlockingError`).

Use case: the support lead adds agents by writing Markdown; each becomes a callable assistant.

Run:  uv run poe server, then: uv run poe lesson 8.9
"""

import argparse

from docshelp.agent_templates import load_templates
from docshelp.config import ensure_llm_ready
from docshelp.console import header
from docshelp.deployment import connect, last_text, server_url

QUESTIONS = {
    "refund-desk": "We were charged twice for March. Can we get a refund?",
    "docs-researcher": "What does the documentation say about offline mode?",
}
ADA = {"name": "Ada", "plan": "team", "account_id": "acct-1001"}
MAX_APPROVALS = 3


def deploy(client, name: str, description: str) -> dict:
    """'Deploying' a template: one API call, no code."""
    return client.assistants.create(
        "agent_from_template",
        config={"configurable": {"template": name}},
        name=name,
        description=description,
        metadata={"source": f"agents/{name}.md"},
    )


def ask(client, assistant_id: str, question: str) -> tuple[str, list[str]]:
    thread_id = client.threads.create()["thread_id"]
    client.runs.wait(thread_id, assistant_id, input={"messages": [{"role": "user", "content": question}]}, context=ADA)
    state, approved = client.threads.get_state(thread_id), []
    while state["next"] and len(approved) < MAX_APPROVALS:  # templates may require approval (`approve:`)
        actions = state["tasks"][0]["interrupts"][0]["value"]["action_requests"]
        approved += [a["name"] for a in actions]
        decisions = [{"type": "approve"} for _ in actions]
        client.runs.wait(thread_id, assistant_id, command={"resume": {"decisions": decisions}}, context=ADA)
        state = client.threads.get_state(thread_id)
    return last_text(state["values"]), approved


def run(client) -> dict:
    templates = load_templates()
    header(f"Templates in agents/: {', '.join(templates)}")
    deployed, answers = {}, {}
    for name, question in QUESTIONS.items():
        assistant = deploy(client, name, templates[name].description)
        deployed[name] = assistant["assistant_id"]
        print(f"\nDeployed {name} as assistant {assistant['assistant_id']}")
        answer, approved = ask(client, assistant["assistant_id"], question)
        answers[name] = answer
        print(f"Customer: {question}")
        if approved:
            print(f"  (approved: {approved})")
        print(f"{name}: {answer}")

    found = client.assistants.search(graph_id="agent_from_template", metadata={"source": "agents/refund-desk.md"})
    print(f"\nAssistants built from agents/refund-desk.md: {len(found)}")
    for assistant_id in deployed.values():
        client.assistants.delete(assistant_id)
    return {"deployed": list(deployed), "answers": answers}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=server_url(), help="the LangGraph Server, e.g. http://127.0.0.1:2024")
    args = parser.parse_args()
    ensure_llm_ready()
    run(connect(args.url))


if __name__ == "__main__":
    main()
