"""Deep Agents for DocsHelp: the Support Ops analyst and the parts it's built from (Part 7).

A *deep agent* is `create_agent` (Part 1) plus a harness that suits long, multi-step work:
  * a virtual filesystem (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`): somewhere to keep
    notes and drafts outside the context window
  * subagents (`task`): hand a self-contained job to a fresh agent and get back only its answer
  * summarization that offloads old history to files instead of dropping it
  * optional planning (`write_todos`), skills, `AGENTS.md` memory, a shell and an interpreter

Everything here runs on any chat model and needs no network: the docs, tickets and reports are local files.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal
from unittest import mock

from deepagents import FilesystemPermission, SubAgent, create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend, StoreBackend
from deepagents.middleware import async_subagents
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, TodoListMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

from docshelp.config import DATA_DIR, DOCS_DIR, PROJECT_ROOT
from docshelp.sources import load_sample_tickets
from docshelp.sources import search_changelog as _search_changelog
from docshelp.tools import make_search_docs

SKILLS_DIR = PROJECT_ROOT / "skills"
REPORTS_DIR = DATA_DIR / "reports"
MEMORY_PATH = "/memories/AGENTS.md"
DEFAULT_MEMORY = """# Support Ops team notes

- Cite the doc file for every fix you mention, e.g. (03-sync-and-offline.md).
- Ticket ids look like NIM-1A2B3C. Quote them exactly.
"""

# Real models, small ones especially, can keep calling tools for a long time: deep agents allow ~10,000 steps.
# Every agent in the lessons gets a budget of model calls per run (Part 2.5's call limits).
MAX_MODEL_CALLS = 8


# --- Tools -----------------------------------------------------------------------------------------------


@tool
def list_tickets(status: Literal["open", "solved", "any"] = "open", platform: str = "") -> str:
    """List support tickets as JSON lines, filtered by status and (optionally) platform, e.g. 'windows'."""
    tickets = [
        t
        for t in load_sample_tickets()
        if status in ("any", t["status"]) and (not platform or t["platform"] == platform.strip().lower())
    ]
    return "\n".join(json.dumps(t) for t in tickets) or "No tickets match."


@tool
def search_changelog(query: str) -> str:
    """Search the Nimbus Notes release notes for fixes and changes related to the query."""
    return "\n".join(_search_changelog(query)) or "Nothing in the changelog matches."


def ops_tools(retriever: BaseRetriever | None = None) -> list[BaseTool]:
    """What the Support Ops analyst works with: the docs, the changelog and the ticket queue."""
    return [make_search_docs(retriever), search_changelog, list_tickets]


def budget(max_model_calls: int = MAX_MODEL_CALLS) -> ModelCallLimitMiddleware:
    """Stop a run after a number of model calls, ending it gracefully rather than with an error."""
    return ModelCallLimitMiddleware(run_limit=max_model_calls, exit_behavior="end")


def general_purpose(llm: BaseChatModel, max_model_calls: int = MAX_MODEL_CALLS) -> SubAgent:
    """The built-in `general-purpose` subagent, with a budget.

    Deep agents add it automatically so the main agent can always delegate. Passing a subagent with the same
    name replaces it, which is how we give it the same model-call budget as its parent.
    """
    return {
        "name": "general-purpose",
        "description": "A general helper for self-contained, multi-step tasks. It has the same tools as you.",
        "system_prompt": "Complete the task you are given, then reply with a concise answer.",
        "model": llm,
        "middleware": [budget(max_model_calls)],
    }


def deep_agent(llm: BaseChatModel, *, tools=(), subagents=(), middleware=(), planning: bool = False, **kwargs):
    """`create_deep_agent` with the lessons' defaults: an explicit model, a budget, and optional planning."""
    return create_deep_agent(
        model=llm,
        tools=list(tools),
        subagents=[general_purpose(llm), *subagents],
        middleware=[*([TodoListMiddleware()] if planning else []), budget(), *middleware],
        **kwargs,
    )


def files_of(state: dict) -> dict[str, str]:
    """The text of every file in a deep agent's state (`StateBackend`), by path."""
    return {path: data["content"] for path, data in state.get("files", {}).items()}


# --- Backends: where the agent's files live -------------------------------------------------------------


def team_memory_namespace(_runtime) -> tuple[str, ...]:
    """The Store namespace for Support Ops memories: shared by every thread and every run of the agent."""
    return ("support-ops", "memories")


DOCS_READ_ONLY = FilesystemPermission(operations=["write"], paths=["/docs/**", "/skills/**"], mode="deny")
APPROVE_REPORTS = FilesystemPermission(operations=["write"], paths=["/reports/**"], mode="interrupt")


def ops_backend(reports_dir: Path | None = None) -> CompositeBackend:
    """One filesystem for the agent, assembled from four backends by path prefix (lesson 7.5).

    /docs/      the product docs on disk            (read-only through `DOCS_READ_ONLY`)
    /skills/    the repo's skills/ folder on disk  (read-only too)
    /memories/  the LangGraph Store                (survives across threads: AGENTS.md lives here)
    /reports/   data/reports/ on disk              (finished reports, for people to read)
    anything else: the agent's state               (scratch notes, gone when the thread is)
    """
    return CompositeBackend(
        default=StateBackend(),
        routes={
            "/docs/": FilesystemBackend(root_dir=DOCS_DIR, virtual_mode=True),
            "/skills/": FilesystemBackend(root_dir=SKILLS_DIR, virtual_mode=True),
            "/memories/": StoreBackend(namespace=team_memory_namespace),
            "/reports/": FilesystemBackend(root_dir=reports_dir or REPORTS_DIR, virtual_mode=True),
        },
    )


def seed_memory(store: BaseStore, text: str = DEFAULT_MEMORY) -> None:
    """Put the team's starting `AGENTS.md` into the Store, unless the agent has already written one."""
    namespace, key = team_memory_namespace(None), MEMORY_PATH.removeprefix("/memories")
    if store.get(namespace, key) is None:
        store.put(namespace, key, {"content": text, "encoding": "utf-8"})


def read_memory(store: BaseStore) -> str:
    item = store.get(team_memory_namespace(None), MEMORY_PATH.removeprefix("/memories"))
    return item.value["content"] if item else ""


# --- A subagent team (lesson 7.11) ------------------------------------------------------------------------


class TicketAnalysis(BaseModel):
    """What the ticket analyst reports back: structured, so the main agent can use it directly."""

    themes: list[str] = Field(description="The recurring problems, most common first.")
    ticket_ids: list[str] = Field(description="The ids of the tickets behind those themes.")
    urgent: bool = Field(description="True if any open ticket is high priority or mentions missing data.")


def subagent_team(retriever: BaseRetriever | None = None, *, fast_llm: BaseChatModel | None = None) -> list[SubAgent]:
    """Specialists the analyst delegates to. Each has its own prompt, tools, and (optionally) model."""
    researcher: SubAgent = {
        "name": "docs-researcher",
        "description": "Finds what the docs and changelog say about a problem: fixes, workarounds, known issues.",
        "system_prompt": (
            "You research Nimbus Notes problems in the docs and changelog. Search a few times, then reply in "
            "at most 120 words, citing the doc file or release for each fact."
        ),
        "tools": [make_search_docs(retriever), search_changelog],
        "middleware": [budget()],
    }
    analyst: SubAgent = {
        "name": "ticket-analyst",
        "description": "Reads the ticket queue and reports recurring themes, the ticket ids, and urgency.",
        "system_prompt": "You analyse the support ticket queue. List the tickets, then report the themes.",
        "tools": [list_tickets],
        "response_format": TicketAnalysis,
        "middleware": [budget()],
    }
    if fast_llm is not None:  # routine work can go to a cheaper, faster model
        analyst["model"] = fast_llm
    return [researcher, analyst]


# --- The capstone: the Support Ops analyst (lesson 7.14) ------------------------------------------------

OPS_ANALYST_PROMPT = """You are the Nimbus Notes Support Ops analyst. You turn the ticket queue into incident reports.

Work like this:
1. Plan with write_todos.
2. Delegate: ask ticket-analyst for the themes in the queue, and docs-researcher what the docs say about them.
3. Use a skill when one fits (read its SKILL.md first).
4. Write the report to /reports/<short-name>.md: summary, affected tickets, cause or workaround, next steps.
5. If you learn a lasting preference from the team, add it to /memories/AGENTS.md with edit_file.
"""


def build_ops_analyst(
    llm: BaseChatModel,
    *,
    retriever: BaseRetriever | None = None,
    fast_llm: BaseChatModel | None = None,
    store: BaseStore | None = None,
    checkpointer=None,
    reports_dir: Path | None = None,
    approve_reports: bool = True,
    on_server: bool = False,
):
    """Every Part 7 feature in one agent: planning, a composite backend, skills, memory, subagents and HITL.

    `on_server=True` leaves persistence to LangGraph Server, which gives every graph its own checkpointer
    and store (Part 8).
    """
    persistence = {}
    if not on_server:
        store = store if store is not None else InMemoryStore()
        seed_memory(store)
        persistence = {"store": store, "checkpointer": checkpointer or InMemorySaver()}
    permissions = [DOCS_READ_ONLY, *([APPROVE_REPORTS] if approve_reports else [])]
    return deep_agent(
        llm,
        tools=ops_tools(retriever),
        subagents=subagent_team(retriever, fast_llm=fast_llm),
        planning=True,
        system_prompt=OPS_ANALYST_PROMPT,
        backend=ops_backend(reports_dir),
        permissions=permissions,
        skills=["/skills/"],
        memory=[MEMORY_PATH],
        name="ops_analyst",
        **persistence,
    )


# --- A local Agent Protocol server, for async subagents (lesson 7.13) -----------------------------------

RESEARCHER_PROMPT = (
    "You research Nimbus Notes problems in the docs and changelog, then answer in at most 80 words with sources."
)


def build_docs_researcher(llm: BaseChatModel, retriever: BaseRetriever | None = None):
    """A plain agent for background research: what an async subagent runs (served as `docs_researcher`)."""
    tools = [make_search_docs(retriever), search_changelog]
    return create_agent(llm, tools, system_prompt=RESEARCHER_PROMPT, middleware=[budget(4)], name="docs_researcher")


def _as_dict(message: BaseMessage | dict) -> dict:
    if isinstance(message, dict):
        return message
    return {"role": message.type, "content": message.text}


class _Threads:
    def __init__(self, server: LocalAgentServer):
        self._server = server

    def create(self, **_kwargs) -> dict:
        thread_id = f"task-{len(self._server.values) + 1}"  # readable ids; a real server uses UUIDs
        self._server.values[thread_id] = {"messages": []}
        return {"thread_id": thread_id}

    def get(self, thread_id: str, **_kwargs) -> dict:
        return {"thread_id": thread_id, "values": self._server.values[thread_id]}


class _Runs:
    def __init__(self, server: LocalAgentServer):
        self._server = server

    def create(self, thread_id: str, assistant_id: str, *, input: dict, **_kwargs) -> dict:  # noqa: A002 - SDK name
        run = {"run_id": f"run-{len(self._server.runs) + 1}", "thread_id": thread_id, "status": "pending"}
        self._server.runs[run["run_id"]] = run
        worker = threading.Thread(target=self._server.execute, args=(run, assistant_id, input), daemon=True)
        self._server.workers.append(worker)
        worker.start()
        return dict(run)

    def get(self, thread_id: str, run_id: str, **_kwargs) -> dict:
        return dict(self._server.runs[run_id])

    def cancel(self, thread_id: str, run_id: str, **_kwargs) -> None:
        self._server.runs[run_id]["status"] = "interrupted"  # its result will be discarded

    def join(self, thread_id: str, run_id: str, **_kwargs) -> dict:
        """Wait for a run to finish and return the thread's values, like the SDK's `runs.join`."""
        self._server.wait()
        return self._server.values[thread_id]


class LocalAgentServer:
    """A tiny, in-process stand-in for an Agent Protocol server such as LangGraph Server (`langgraph dev`).

    It implements just the client calls async subagents make: create a thread, start a run in the
    background, poll its status, read the thread's messages, and cancel. Each run executes a graph in a
    worker thread, so the calling agent carries on while it works, exactly as with a real server.
    """

    def __init__(self, graphs: dict):
        self.graphs = graphs
        self.values: dict[str, dict] = {}
        self.runs: dict[str, dict] = {}
        self.workers: list[threading.Thread] = []
        self.threads = _Threads(self)
        self.runs_api = _Runs(self)

    def execute(self, run: dict, assistant_id: str, run_input: dict) -> None:
        run["status"] = "running"
        history = self.values[run["thread_id"]]["messages"]
        try:  # like a server's checkpointer, the thread keeps the conversation across runs
            result = self.graphs[assistant_id].invoke({"messages": [*history, *run_input["messages"]]})
        except Exception as err:  # noqa: BLE001 - reported to the caller as the run's error, like a server
            run.update(status="error", error=f"{type(err).__name__}: {err}")
            return
        if run["status"] != "interrupted":  # cancelled runs leave the thread as it was
            self.values[run["thread_id"]] = {"messages": [_as_dict(m) for m in result["messages"]]}
            run["status"] = "success"

    def wait(self, timeout: float = 600) -> None:
        """Block until every background run has finished (handy in scripts and tests)."""
        for worker in self.workers:
            worker.join(timeout)


class _SyncClient:
    """The shape of `langgraph_sdk.get_sync_client()`: `.threads` and `.runs`."""

    def __init__(self, server: LocalAgentServer):
        self.threads = server.threads
        self.runs = server.runs_api


@contextmanager
def serve_locally(server: LocalAgentServer) -> Iterator[LocalAgentServer]:
    """Route async subagents to `server` instead of a URL, by swapping in its client.

    With a real server, skip this and give each `AsyncSubAgent` a `url`, e.g. http://127.0.0.1:2024.
    """
    client = _SyncClient(server)
    with mock.patch.object(async_subagents, "get_sync_client", lambda **_kwargs: client):
        yield server
