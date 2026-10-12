"""Shipping it: from `langgraph dev` to a production deployment (Docker image, self-hosted, or cloud).

Academy: Introduction to LangGraph M6 L2 "Creating a Deployment"; Introduction to LangSmith Deployment M1
"Deploy"; LangSmith Essentials M3 "Deployments".

Concepts:
  * `langgraph dev` is for development: in-memory persistence, one process, hot reload. Production runs the
    same `langgraph.json` in the `langchain/langgraph-api` image with:
      - Postgres (`POSTGRES_URI`): checkpoints, threads, runs, the Store, crons
      - Redis (`REDIS_URI`): streams runs between API servers and queue workers, so both can scale out
      - your secrets as environment variables, never baked into the image
  * `langgraph dockerfile <path>` writes that image's Dockerfile (this lesson does it; no Docker needed);
    `langgraph build -t docshelp-agent` builds it; `langgraph up` starts it with Postgres and Redis. The
    image runs Agent Server, which LangSmith licenses: check the self-hosting terms before running it.
  * LangSmith Deployment (cloud, paid plans): New Deployment -> connect the GitHub repo -> path to
    `langgraph.json` -> environment variables and secrets -> each push builds a revision. You get a URL and
    use a LangSmith API key as `api_key=` in the SDK: the code of lessons 8.1-8.12 is otherwise unchanged.
  * this repo's `compose.yaml` runs `langgraph dev` in containers (a dev environment that works the same on
    every OS), with services `server` (port 2024), `server-auth` (2025) and an optional `ollama`.
  * production checklist: health checks (`/ok`, and `/health` from 8.6), auth on (8.10), CORS limited to
    your UI's origin, no blocking calls in graphs (8.9), budgets on agents (Part 7), tracing to LangSmith
    (Part 5) and monitoring (Part 9).

Run:  uv run poe server, then: uv run poe lesson 8.13
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from docshelp.config import PROJECT_ROOT, ensure_llm_ready
from docshelp.console import header
from docshelp.deployment import connect, server_url


def langgraph_cli() -> str:
    """The `langgraph` command installed next to this Python (uv's environment), else from PATH."""
    scripts = Path(sys.executable).parent
    for name in ("langgraph", "langgraph.exe"):
        if (scripts / name).exists():
            return str(scripts / name)
    return shutil.which("langgraph") or "langgraph"


def generate_dockerfile(target: Path, config: Path = PROJECT_ROOT / "langgraph.json") -> str:
    subprocess.run(
        [langgraph_cli(), "dockerfile", "-c", str(config), str(target)],
        check=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,  # never wait for input
        timeout=120,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return target.read_text(encoding="utf-8")


def run(client, url: str, workdir: Path | None = None) -> dict:
    header("1. The production image: `langgraph dockerfile`")
    with tempfile.TemporaryDirectory() as tmp:
        dockerfile = generate_dockerfile(Path(workdir or tmp) / "Dockerfile")
    important = [line for line in dockerfile.splitlines() if line.startswith(("FROM", "ADD", "ENV"))]
    for line in important:
        print(f"  {line[:110]}{' ...' if len(line) > 110 else ''}")
    print("The base image is Agent Server; your graphs, routes and store config arrive as environment variables.")

    header("2. This repo's containers: compose.yaml")
    services = yaml.safe_load((PROJECT_ROOT / "compose.yaml").read_text(encoding="utf-8"))["services"]
    for name, service in services.items():
        ports = ", ".join(service.get("ports", [])) or "-"
        print(f"  {name:12} ports {ports:12} {' '.join(map(str, service.get('command', [])))[:70]}")

    header("3. Same API wherever it runs")
    graphs = sorted({a["graph_id"] for a in client.assistants.search(limit=100)})
    print(f"{url} serves {len(graphs)} graphs. Point the SDK at a cloud URL with api_key=... and nothing else changes.")
    return {"dockerfile": important, "services": list(services), "graphs": graphs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=server_url(), help="the LangGraph Server, e.g. http://127.0.0.1:2024")
    args = parser.parse_args()
    ensure_llm_ready()
    run(connect(args.url), args.url)


if __name__ == "__main__":
    main()
