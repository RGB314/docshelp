"""Middleware: hook into every step of the agent loop.

Academy: Intro to LangChain M3 L1 "What is Middleware?".

Concepts:
  * the six hooks: `before_agent`, `before_model`, `after_model`, `after_agent` (node-style: run once, may
    update state) and `wrap_model_call`, `wrap_tool_call` (wrap each call: time it, change it, skip it)
  * class-based middleware (`AgentMiddleware` subclass, `src/docshelp/middleware.py::AuditLog`) for
    hooks that share configuration or state; decorator-based (`@before_agent`, ...) for one hook
  * jumping: a hook declared with `can_jump_to=["end"]` can stop the run early by returning
    `{"jump_to": "end"}`. Here a guard refuses messages that contain a password, before any model call.
  * ordering: `before_*` hooks run in list order, `after_*` in reverse, wrap hooks nest
  * you already used one: `@dynamic_prompt` (lesson 1.9) is middleware wrapping each model call

Run:  uv run poe lesson 2.4     (the audit log is written to data/audit.jsonl)
"""

import re
from pathlib import Path

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever

from docshelp.config import DATA_DIR, describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.ingest import get_retriever
from docshelp.middleware import AuditLog
from docshelp.tools import make_search_docs

SECRET = re.compile(r"\b(password|passcode|api[_ ]?key)\b\s*[:=]", re.IGNORECASE)


@before_agent(can_jump_to=["end"])
def refuse_secrets(state: AgentState, runtime) -> dict | None:
    """Stop before the model ever sees a message that contains a credential."""
    if SECRET.search(state["messages"][-1].text):
        reply = "Please never share passwords or keys with support. Change that password now; we never need it."
        return {"messages": [AIMessage(reply)], "jump_to": "end"}
    return None


def run(llm: BaseChatModel, retriever: BaseRetriever, log_path: Path | None = None) -> dict:
    audit = AuditLog(log_path)
    agent = create_agent(
        llm,
        tools=[make_search_docs(retriever)],
        system_prompt="You are DocsHelp, the Nimbus Notes support assistant. Search the docs before answering.",
        middleware=[refuse_secrets, audit],
    )

    header("1. A normal question: every hook fires")
    question = "How do I export a notebook as PDF?"
    print(f"User: {question}")
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print(f"DocsHelp: {result['messages'][-1].text}\n")
    events = [e["event"] for e in audit.events]
    for event in audit.events:
        details = {k: v for k, v in event.items() if k not in ("event", "at")}
        print(f"  {event['event']:<13} {details}")

    header("2. A message with a password: the guard ends the run early")
    question = "I can't log in. My password: hunter2. Can you check my account?"
    print(f"User: {question}")
    blocked = agent.invoke({"messages": [{"role": "user", "content": question}]})["messages"][-1].text
    print(f"DocsHelp: {blocked}")
    if log_path:
        print(f"\nAudit log: {log_path}")
    return {"events": events, "blocked": blocked}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever(), DATA_DIR / "audit.jsonl")


if __name__ == "__main__":
    main()
