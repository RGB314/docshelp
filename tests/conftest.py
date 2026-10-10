"""Offline test helpers: scripted fake chat models and a fake-embedding retriever (no API keys, no downloads)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import Field

from docshelp.ingest import load_docs, split_docs
from docshelp.tasks import load_lesson


class ScriptedLLM:
    """Stands in for a chat model. Returns pre-scripted responses in order.

    `structured` maps a Pydantic schema to the list of objects `with_structured_output(schema)`
    should return; `text` is the list of strings plain `invoke` calls return.
    """

    def __init__(self, structured: dict[type, list] | None = None, text: list[str] | None = None):
        self.structured = {k: list(v) for k, v in (structured or {}).items()}
        self.text = list(text or [])

    def with_structured_output(self, schema, **_):
        return RunnableLambda(lambda _input: self.structured[schema].pop(0))

    def invoke(self, _messages, config=None):
        return AIMessage(self.text.pop(0))


USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


class FakeChatModel(BaseChatModel):
    """A real LangChain chat model with scripted replies.

    Unlike `ScriptedLLM` it is a full `BaseChatModel`, so it works inside LCEL chains, `create_agent`
    and tool-calling loops, and it streams word by word. `replies` are returned in order (a str, or an
    `AIMessage` with `tool_calls`); `structured` works like `ScriptedLLM`'s. Every list of messages the
    model receives is recorded in `seen`, so tests can check prompts.
    """

    replies: list[Any] = Field(default_factory=list)
    structured: dict[Any, list] = Field(default_factory=dict)
    seen: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def _next(self, messages: list[BaseMessage]) -> AIMessage:
        self.seen.append(list(messages))
        reply = self.replies.pop(0)
        reply = AIMessage(reply) if isinstance(reply, str) else reply
        return reply.model_copy(update={"usage_metadata": reply.usage_metadata or USAGE})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        reply = self._next(messages)
        words = reply.text.split(" ") if reply.text else [""]
        for i, word in enumerate(words):
            chunk = AIMessageChunk(content=word if i == 0 else f" {word}", id=reply.id)
            if i == len(words) - 1:
                chunk = chunk + AIMessageChunk(
                    content="",
                    usage_metadata=reply.usage_metadata,
                    tool_call_chunks=[
                        tool_call_chunk(name=c["name"], args=json.dumps(c["args"]), id=c["id"], index=n)
                        for n, c in enumerate(reply.tool_calls)
                    ],
                )
            yield ChatGenerationChunk(message=chunk)

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(lambda _input: self.structured[schema].pop(0))


def tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    """An AI message that asks to call one tool: the shape every provider's model returns."""
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


@pytest.fixture(scope="session")
def fake_retriever():
    store = InMemoryVectorStore(embedding=DeterministicFakeEmbedding(size=64))
    store.add_documents(split_docs(load_docs()))
    return store.as_retriever(search_kwargs={"k": 3})


@pytest.fixture
def lesson():
    """`lesson("1.6")` imports that lesson's module (lesson files start with digits, so not importable by name)."""
    return load_lesson
