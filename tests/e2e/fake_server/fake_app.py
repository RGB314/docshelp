"""The real DocsHelp server graphs, served by LangGraph Server in E2E tests with a rule-based fake model.

Everything except the model, the embeddings and the data paths is real: `docshelp.server`'s graphs, the
custom routes, auth, the Store index, checkpointing, interrupts, the server API and the SDK client. The fake
model is deterministic and needs no script per test: on each call it uses the next "canned" tool it is offered
and hasn't used yet in the conversation, then answers in text. That drives every path the Part 8 lessons show
(tool calls, HITL interrupts, memory writes, file writes) without network or keys.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import Field

from docshelp import config, ingest

DATA = Path(os.environ["DOCSHELP_E2E_DATA"])  # a temp dir per test session
DELAY = float(os.getenv("DOCSHELP_E2E_FAKE_DELAY", "0"))  # seconds per model call, so runs overlap (8.4)

CANNED: dict[str, dict] = {  # tool -> arguments, in the order the fake model uses them
    "search_docs": {"query": "sync"},
    "UpdateMemory": {"update_type": "profile"},
    "CustomerProfile": {"name": "Ada", "operating_systems": ["Windows"], "devices": ["iPad"]},
    "write_file": {"file_path": "/reports/sync-incident.md", "content": "# Sync incident\n\nTL;DR: fake report."},
    "edit_file": {
        "file_path": "/memories/AGENTS.md",
        "old_string": "# Support Ops team notes",
        "new_string": "# Support Ops team notes\n\n- Start every report with a one-line TL;DR.",
    },
    "send_reply": {"to": "customer@example.com", "subject": "Re: your question", "body": "Thanks for writing in."},
    "done": {},
}


def last_human(messages) -> str:
    return next((m.text for m in reversed(messages) if isinstance(m, HumanMessage)), "")


def structured(schema, messages) -> object:
    text = last_human(messages) if isinstance(messages, list) else str(messages)
    name = schema.__name__ if isinstance(schema, type) else schema.get("title")
    canned = {
        "Route": {"intent": "ticket_request" if "ticket" in text.lower() else "docs_question", "standalone_question": text},
        "Grade": {"relevant": True, "reason": "The docs cover it."},
        "TicketDraft": {"title": "Webhooks not delivered", "description": text, "priority": "high"},
        "RouterSchema": {"reasoning": "A customer question.", "classification": "respond"},
    }[name]
    return schema(**canned) if isinstance(schema, type) else canned


class RuleModel(BaseChatModel):
    tools: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "rule-fake"

    def _reply(self, messages) -> AIMessage:
        time.sleep(DELAY)
        used = {c["name"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls}
        for name, args in CANNED.items():
            if name in self.tools and name not in used and (name != "done" or "send_reply" in used):
                call = {"name": name, "args": args, "id": f"call_{name}_{len(messages)}", "type": "tool_call"}
                return AIMessage("", tool_calls=[call])
        return AIMessage(f"Nimbus Notes support here. You asked: {last_human(messages)[:80]}")

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._reply(messages))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        """Word by word, like a real model, so token streaming (8.3, 8.11) has tokens to show."""
        reply = self._reply(messages)
        if reply.tool_calls:
            chunks = [tool_call_chunk(name=c["name"], args=json.dumps(c["args"]), id=c["id"], index=0) for c in reply.tool_calls]
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=chunks))
            return
        for i, word in enumerate(reply.text.split(" ")):
            yield ChatGenerationChunk(message=AIMessageChunk(content=word if i == 0 else f" {word}"))

    def bind_tools(self, tools, **kwargs):
        names = [t.name if isinstance(t, BaseTool) else convert_to_openai_tool(t)["function"]["name"] for t in tools]
        return self.model_copy(update={"tools": names})

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(lambda messages: structured(schema, messages))


def fake_llm(*args, **kwargs) -> RuleModel:
    return RuleModel()


def fake_vector_store():
    store = InMemoryVectorStore(embedding=DeterministicFakeEmbedding(size=64))
    store.add_documents(ingest.split_docs(ingest.load_docs()))
    return store


# Patch before docshelp.server is imported, so every module picks up the fakes.
config.get_llm = fake_llm
ingest.build_vector_store = fake_vector_store

from docshelp import deep, tools  # noqa: E402

tools.TICKETS_PATH = DATA / "tickets.jsonl"
tools.OUTBOX_PATH = DATA / "outbox.jsonl"
deep.REPORTS_DIR = DATA / "reports"

from docshelp.auth import auth  # noqa: E402, F401
from docshelp.server import *  # noqa: E402, F403 - every served graph, under the name langgraph.json uses
from docshelp.webapp import app  # noqa: E402, F401

_fake_embeddings = DeterministicFakeEmbedding(size=384)


def embed(texts: list[str]) -> list[list[float]]:
    return _fake_embeddings.embed_documents(texts)
