"""Custom routes: your own HTTP endpoints, served by the same deployment as the agent API.

Academy: Introduction to LangSmith Deployment M2 "Custom Routes".

Concepts:
  * `langgraph.json` -> `"http": {"app": "./src/docshelp/webapp.py:app"}` mounts any ASGI app (Starlette,
    FastAPI) into the server: same process, same port, same deployment. See `docshelp/webapp.py`.
  * typical uses: a health check for load balancers, small read APIs for other systems (here, the tickets
    the agents opened), webhooks receivers, and web pages (lesson 8.11 serves a chat page at /ui)
  * your routes take precedence over the server's when paths clash, and they show up in the server's
    OpenAPI docs (`/docs`). The `http` section also configures CORS (`"cors": {...}`), and can switch off
    built-in parts of the API (`"disable_assistants"`, `"disable_ui"`, ...).
  * auth: custom routes are public unless `"enable_custom_route_auth": true` (lesson 8.10)

Use case: the help-desk dashboard reads the tickets DocsHelp opened; the load balancer polls /health.

Run:  uv run poe server, then: uv run poe lesson 8.6
"""

import argparse

import httpx

from docshelp.config import ensure_llm_ready
from docshelp.console import header
from docshelp.deployment import connect, server_url


def run(client, url: str) -> dict:
    header("GET /health")
    health = httpx.get(f"{url}/health", timeout=10).json()
    print(health)

    header("GET /tickets and /tickets?priority=high")
    every = httpx.get(f"{url}/tickets", timeout=10).json()
    high = httpx.get(f"{url}/tickets", params={"priority": "high"}, timeout=10).json()
    print(f"{every['count']} tickets, {high['count']} high priority")
    for ticket in every["tickets"][-3:]:
        print(f"  {ticket.get('id', '?')}  [{ticket.get('priority')}] {ticket.get('title') or ticket.get('summary')}")
    if not every["count"]:
        print("  (none yet: lesson 8.3 opens one)")

    header("GET /ui: a web page from the same server (lesson 8.11)")
    page = httpx.get(f"{url}/ui", timeout=10)
    print(f"{page.status_code} {page.headers['content-type']}, {len(page.text):,} characters: open {url}/ui")

    header("The routes sit next to the agent API, in one OpenAPI document")
    paths = httpx.get(f"{url}/openapi.json", timeout=10).json()["paths"]
    ours = sorted(p for p in paths if p in ("/health", "/tickets", "/ui"))
    print(f"Ours: {ours}; the server's own include {[p for p in sorted(paths) if p.startswith('/threads')][:3]} ...")
    agent_still_works = bool(client.assistants.search(limit=1))
    return {"health": health, "tickets": every["count"], "high": high["count"], "ui": page.status_code,
            "documented": ours, "api_ok": agent_still_works}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=server_url(), help="the LangGraph Server, e.g. http://127.0.0.1:2024")
    args = parser.parse_args()
    ensure_llm_ready()
    run(connect(args.url), args.url)


if __name__ == "__main__":
    main()
