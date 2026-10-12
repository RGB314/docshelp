"""DocsHelp as a deployed service: the served support agent, the store's embeddings and client helpers (Part 8).

LangGraph Server (`uv run poe server`, i.e. `langgraph dev`) wraps the graphs listed in `langgraph.json` in an
HTTP API. Lessons in Part 8 are clients of that API. This module holds what both sides share:

  * `build_deployed_agent`: the support agent the server serves as `support_agent`. Its runtime context
    (`AssistantContext`) carries the customer (from each run) and the assistant's settings (from the assistant),
    so one graph can be deployed as several assistants (lesson 8.5).
  * `build_inbox_ingest`: the graph a cron runs to feed new emails to the deployed inbox agent (lesson 8.12).
  * `embed`: the function `langgraph.json` names under `store.index`, so the server's Store can search
    memories by meaning with the same local fastembed model as Part 1's RAG (lesson 8.7).
  * `server_url` and `connect`: how lessons find the server and fail with a helpful message when it's down.
"""

from __future__ import annotations

import asyncio
import os
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, dynamic_prompt
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import START, StateGraph
from langgraph_sdk import get_client

from docshelp.agents import support_prompt
from docshelp.ambient import load_inbox, public
from docshelp.customers import Customer
from docshelp.tools import make_search_docs, my_account, quote_price

DEFAULT_URL = "http://127.0.0.1:2024"  # `uv run poe server`
DEFAULT_AUTH_URL = "http://127.0.0.1:2025"  # `uv run poe server-auth` (lesson 8.10)


@dataclass
class AssistantContext(Customer):
    """Runtime context of the served support agent.

    The customer fields (name, plan, account_id) come with each run. `style` and `model` are the assistant's
    settings: an assistant stores them once, and the server merges them into every run it starts.
    """

    style: Literal["concise", "detailed"] = "concise"
    model: Literal["default", "fast"] = "default"


STYLE_GUIDANCE = {
    "concise": "Answer in at most three sentences.",
    "detailed": "Answer step by step with numbered instructions, and end with a tip.",
}


def build_deployed_agent(
    llm: BaseChatModel, fast_llm: BaseChatModel | None = None, retriever: BaseRetriever | None = None
):
    """The support agent as served (`support_agent`): Part 1's agent with per-assistant style and model.

    No checkpointer and no store: on the server, every graph gets the server's own.
    """

    @dynamic_prompt
    def styled_prompt(request: ModelRequest) -> str:
        context = request.runtime.context or AssistantContext()
        return f"{support_prompt(context)} {STYLE_GUIDANCE[context.style]}"

    class PickModel(AgentMiddleware):
        """Dynamic model selection (lesson 2.8), in both flavours: the server runs graphs asynchronously."""

        def choose(self, request: ModelRequest) -> ModelRequest:
            context = request.runtime.context or AssistantContext()
            return request.override(model=fast_llm) if context.model == "fast" and fast_llm is not None else request

        def wrap_model_call(self, request, handler):
            return handler(self.choose(request))

        async def awrap_model_call(self, request, handler):
            return await handler(self.choose(request))

    return create_agent(
        llm,
        tools=[make_search_docs(retriever), quote_price, my_account],
        middleware=[styled_prompt, PickModel()],
        context_schema=AssistantContext,
        name="support_agent",
    )


# --- The ambient inbox, deployed (lesson 8.12) --------------------------------------------------------------

INBOX_THREADS = uuid.UUID("6f1c3c4e-2b1a-4d7e-9a53-1e0c8a4f9b21")  # any fixed UUID: the namespace for thread ids


def email_thread_id(email_id: str) -> str:
    """The same email always maps to the same thread, so ingesting twice never answers twice."""
    return str(uuid.uuid5(INBOX_THREADS, email_id))


class IngestState(TypedDict, total=False):
    limit: int  # how many emails to take from the inbox (0 or missing: all)
    started: list[str]  # the email ids this run handed to the inbox agent


def build_inbox_ingest(load_emails: Callable[[], list[dict]] = load_inbox):
    """The graph a cron runs: fetch new emails and start one background `inbox_agent` run per email.

    It talks to its own server with `get_client()` and no URL: inside LangGraph Server that's a loopback
    client (no network hop, no extra auth). A real app would fetch from Gmail or a help desk API here.
    """

    async def ingest(state: IngestState) -> dict:
        client = get_client()
        emails = await asyncio.to_thread(load_emails)
        started = []
        for email in emails[: state.get("limit") or None]:
            thread_id = email_thread_id(email["id"])
            metadata = {"source": "inbox", "email_id": email["id"], "subject": email["subject"]}
            await client.threads.create(thread_id=thread_id, if_exists="do_nothing", metadata=metadata)
            if await client.runs.list(thread_id, limit=1):  # already handled
                continue
            await client.runs.create(thread_id, "inbox_agent", input={"email": public(email)})
            started.append(email["id"])
        return {"started": started}

    builder = StateGraph(IngestState)
    builder.add_node(ingest)
    builder.add_edge(START, "ingest")
    return builder.compile(name="inbox_ingest")


# --- The Store's semantic index -----------------------------------------------------------------------------

EMBEDDING_DIMS = 384  # BAAI/bge-small-en-v1.5, the default local embedding model (see `Settings`)


@lru_cache(maxsize=1)
def _embeddings():
    from docshelp.config import get_embeddings

    return get_embeddings()


def embed(texts: list[str]) -> list[list[float]]:
    """Embed texts for the server's Store index (`langgraph.json` -> `store.index.embed`).

    The server calls this whenever an indexed item is put, and for each `search_items(query=...)`.
    The model loads on first use, so a server that never searches memories never loads it.
    """
    return _embeddings().embed_documents(texts)


# --- Client side ------------------------------------------------------------------------------------------


def server_url(default: str = DEFAULT_URL, var: str = "LANGGRAPH_URL") -> str:
    """Where the server is: `LANGGRAPH_URL` (Docker Compose sets it to the `server` service), else localhost."""
    return os.getenv(var) or default


def connect(url: str, *, headers: dict | None = None, start: str = "uv run poe server"):
    """A sync SDK client for `url`, or exit with how to start the server if nothing answers there."""
    from langgraph_sdk import get_sync_client

    try:
        urllib.request.urlopen(f"{url}/ok", timeout=5)
    except OSError as err:
        raise SystemExit(
            f"Can't reach LangGraph Server at {url}. Start it first with `{start}` "
            f"(Docker: `docker compose up -d server`), or pass --url.\n({err})"
        ) from None
    return get_sync_client(url=url, headers=headers)


def last_text(state: dict) -> str:
    """The text of the last message in a thread's values (messages arrive as dicts over the API)."""
    messages = state.get("messages") or [{}]
    content = messages[-1].get("content", "")
    if isinstance(content, list):  # some providers return content blocks
        content = " ".join(block.get("text", "") for block in content if isinstance(block, dict))
    return content
