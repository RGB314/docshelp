"""Offline test helpers: a scripted fake LLM and a fake-embedding retriever (no API keys, no downloads)."""

from __future__ import annotations

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.vectorstores import InMemoryVectorStore

from docshelp.ingest import load_docs, split_docs


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


@pytest.fixture(scope="session")
def fake_retriever():
    store = InMemoryVectorStore(embedding=DeterministicFakeEmbedding(size=64))
    store.add_documents(split_docs(load_docs()))
    return store.as_retriever(search_kwargs={"k": 3})
