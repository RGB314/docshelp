"""Deployment concepts: what LangGraph Server adds to a graph, and what `langgraph.json` declares.

Academy: Introduction to LangGraph M1 L8 "Intro to Deployment", M6 L1 "Deployment Concepts";
Introduction to LangSmith Deployment M1 "Concepts", "The Agent", "Deploy"; LangSmith Essentials M3 "Deployments".

Concepts:
  * a graph is a library; a *deployment* is a service. LangGraph Server (the Agent Server) wraps the graphs in
    `langgraph.json` in an HTTP API with:
      - assistants: a graph plus configuration (one default assistant per graph; more in 8.5)
      - threads: conversations whose state the server checkpoints (Part 3's checkpointer, managed for you)
      - runs: one execution on a thread, foreground or background, streamed or waited on (8.3)
      - a task queue and workers, so runs survive slow models and many users (8.4)
      - the Store for long-term memory (8.7), crons (8.12), webhooks, custom routes (8.6) and auth (8.10)
  * `langgraph.json` is the deployment's manifest:
      - `graphs`: graph id -> `path.py:variable` (a compiled graph, or a factory function: 8.9)
      - `dependencies`: what to install (`.` = this project, via pyproject.toml)
      - `env`: the environment file; `python_version`
      - `http.app`: custom routes; `store.index`: semantic search; `auth.path`: auth handlers
  * three ways to run the same `langgraph.json`, same API every time:
      - `langgraph dev` (`uv run poe server`): in-memory, hot reload, Studio. For development; free.
      - `langgraph build` / `langgraph up`: a Docker image with Postgres (checkpoints, store) and Redis
        (streaming between workers): self-hosted production (lesson 8.13).
      - LangSmith Deployment (cloud): connect a GitHub repo, pick `langgraph.json`, set secrets, and each
        revision builds and deploys behind a URL with an API key. It needs a paid LangSmith plan, so this
        course stays on your machine; the client code is identical, only the URL and key change.
  * "the agent" deployed throughout Part 8 is DocsHelp: Part 3's support graph (`docshelp`) and the
    support agent with assistant settings (`support_agent`), plus the agents of Parts 4, 6 and 7.

Run:  uv run poe server            (terminal 1; Docker: docker compose up -d server)
      uv run poe lesson 8.1        (terminal 2)
"""

import argparse
import json
from pathlib import Path

import httpx

from docshelp.config import PROJECT_ROOT, ensure_llm_ready
from docshelp.console import header
from docshelp.deployment import connect, server_url

MANIFEST_KEYS = {
    "dependencies": "packages to install into the server image",
    "graphs": "graph id -> the code that builds it",
    "http": "custom routes mounted next to the API",
    "store": "the long-term memory store: semantic index",
    "auth": "authentication and authorization handlers",
    "env": "environment file loaded into the server",
    "python_version": "the server's Python",
}


def run(client, url: str, manifest: Path = PROJECT_ROOT / "langgraph.json") -> dict:
    config = json.loads(manifest.read_text(encoding="utf-8"))
    header(f"{manifest.name}: the deployment manifest")
    for key, value in config.items():
        if key in MANIFEST_KEYS:
            print(f"  {key:15} {MANIFEST_KEYS[key]}")
    for graph_id, spec in config["graphs"].items():
        print(f"    {graph_id:20} {spec}")

    info = httpx.get(f"{url}/info", timeout=10).json()
    header("The running server")
    print(f"  LangGraph API {info.get('version')}, features: {', '.join(k for k, v in info.get('flags', {}).items() if v)}")

    assistants = client.assistants.search(limit=100)
    header("Assistants: the server made one per graph")
    for assistant in sorted(assistants, key=lambda a: a["graph_id"]):
        print(f"  {assistant['graph_id']:20} assistant {assistant['assistant_id']}")

    drawing = client.assistants.get_graph("docshelp")
    nodes = [n["id"] for n in drawing["nodes"] if not n["id"].startswith("__")]
    print(f"\nThe docshelp graph, as the server sees it: {' -> '.join(nodes)}")
    support = next(a for a in assistants if a["graph_id"] == "support_agent")
    schemas = client.assistants.get_schemas(support["assistant_id"])  # this call needs the id, not the graph name
    context_fields = list((schemas.get("context_schema") or {}).get("properties", {}))
    print(f"support_agent's context schema (what each assistant or run may set): {context_fields}")
    return {
        "graphs": list(config["graphs"]),
        "assistants": {a["graph_id"] for a in assistants},
        "version": info.get("version"),
        "nodes": nodes,
        "context_fields": context_fields,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=server_url(), help="the LangGraph Server, e.g. http://127.0.0.1:2024")
    args = parser.parse_args()
    ensure_llm_ready()  # the server uses the same .env: check the model setup before calling it
    run(connect(args.url), args.url)


if __name__ == "__main__":
    main()
