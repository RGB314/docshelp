"""Custom agent middleware for DocsHelp (Part 2.4).

Middleware hooks into `create_agent`'s loop without changing it. Two kinds of hook:

  * node-style hooks run once at a point in the loop and may return a state update:
      before_agent → (before_model → model → after_model → tools)* → after_agent
  * wrap-style hooks wrap each model or tool call like a decorator, so they can time it, retry it,
    change the request (`request.override(...)`), or skip calling `handler` altogether.

With several middlewares, `before_*` hooks run in list order, `after_*` hooks in reverse, and wrap hooks nest
(the first is outermost). Async agents (`ainvoke`, e.g. with MCP tools) need the `awrap_*` variants.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage


class AuditLog(AgentMiddleware):
    """Records every step of an agent run: an audit trail for support conversations.

    Events go to `self.events` and, if `path` is given, to a JSON-lines file (one object per line).
    """

    def __init__(self, path: Path | None = None):
        super().__init__()
        self.path = path
        self.events: list[dict[str, Any]] = []

    def log(self, event: str, **data: Any) -> None:
        record = {"event": event, "at": time.time(), **data}
        self.events.append(record)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")

    # Node-style hooks: (state, runtime) -> state update or None
    def before_agent(self, state: AgentState, runtime) -> None:
        self.log("before_agent", question=state["messages"][-1].text)

    def before_model(self, state: AgentState, runtime) -> None:
        self.log("before_model", messages=len(state["messages"]))

    def after_model(self, state: AgentState, runtime) -> None:
        last = state["messages"][-1]
        calls = [c["name"] for c in last.tool_calls] if isinstance(last, AIMessage) else []
        usage = getattr(last, "usage_metadata", None) or {}
        self.log("after_model", tool_calls=calls, tokens=usage.get("total_tokens"))

    def after_agent(self, state: AgentState, runtime) -> None:
        self.log("after_agent", answer=state["messages"][-1].text)

    # Wrap-style hooks: (request, handler) -> response. Here: time each call.
    def wrap_model_call(self, request, handler):
        start = time.perf_counter()
        response = handler(request)
        self.log("model", ms=round((time.perf_counter() - start) * 1000))
        return response

    async def awrap_model_call(self, request, handler):
        start = time.perf_counter()
        response = await handler(request)
        self.log("model", ms=round((time.perf_counter() - start) * 1000))
        return response

    def wrap_tool_call(self, request, handler):
        start = time.perf_counter()
        result = handler(request)
        self.log("tool", name=request.tool_call["name"], ms=round((time.perf_counter() - start) * 1000))
        return result

    async def awrap_tool_call(self, request, handler):
        start = time.perf_counter()
        result = await handler(request)
        self.log("tool", name=request.tool_call["name"], ms=round((time.perf_counter() - start) * 1000))
        return result
