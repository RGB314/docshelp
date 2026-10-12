"""Custom routes served by LangGraph Server next to its own API (lesson 8.6), named in `langgraph.json`:

    "http": {"app": "./src/docshelp/webapp.py:app"}

The server mounts this Starlette app into the same process and port as the agent API, so one deployment can
also serve a health check for the load balancer, small endpoints for other systems, and a web page. Any ASGI
app works (FastAPI too); Starlette is used because LangGraph Server already depends on it.

  GET /health    liveness and what's deployed: for load balancers and uptime checks
  GET /tickets   the tickets the agents opened (data/tickets.jsonl), e.g. for a help-desk dashboard
  GET /ui        a tiny chat page that talks to the agent API directly (lesson 8.11, "bring your own UI")

Custom routes skip the server's auth unless `"http": {"enable_custom_route_auth": true}`: then `/tickets`
would see the caller as `request.user` (lesson 8.10).
"""

from __future__ import annotations

import json
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from docshelp import tools

CHAT_PAGE = Path(__file__).with_name("chat.html")


def read_tickets() -> list[dict]:
    path = tools.TICKETS_PATH  # read at request time, so tests can point it elsewhere
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "docshelp", "tickets": len(read_tickets())})


async def tickets(request: Request) -> JSONResponse:
    """List tickets. Filters are plain query parameters, e.g. /tickets?priority=high"""
    found = read_tickets()
    if priority := request.query_params.get("priority"):
        found = [t for t in found if t.get("priority") == priority]
    return JSONResponse({"tickets": found, "count": len(found)})


async def chat_page(request: Request) -> FileResponse:
    return FileResponse(CHAT_PAGE, media_type="text/html")


app = Starlette(
    routes=[
        Route("/health", health),
        Route("/tickets", tickets),
        Route("/ui", chat_page),
    ]
)
