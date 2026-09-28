"""The real DocsHelp graph with a scripted model and fake embeddings, served by LangGraph Server in E2E tests.

Everything except the LLM and the embedding model is real: graph, checkpointing, interrupts, the server
API and the SDK client. The responses are scripted for the two turns stage 5 sends.
"""

import os
from pathlib import Path

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.vectorstores import InMemoryVectorStore

from docshelp import tools
from docshelp.graph import Grade, Route, TicketDraft, build_graph
from docshelp.ingest import load_docs, split_docs

if tickets := os.getenv("DOCSHELP_E2E_TICKETS"):
    tools.TICKETS_PATH = Path(tickets)


class ScriptedLLM:
    def __init__(self):
        self.structured = {
            Route: [
                Route(intent="docs_question", standalone_question="What are the Team plan API rate limits?"),
                Route(intent="ticket_request", standalone_question="Webhooks stopped arriving"),
            ],
            Grade: [Grade(relevant=True, reason="The API doc lists rate limits.")],
            TicketDraft: [
                TicketDraft(title="Webhooks not delivered", description="Since yesterday", priority="high")
            ],
        }
        self.text = ["600 requests per minute [08-api.md]"]

    def with_structured_output(self, schema, **_):
        return RunnableLambda(lambda _input: self.structured[schema].pop(0))

    def invoke(self, _messages, config=None):
        return AIMessage(self.text.pop(0))


store = InMemoryVectorStore(embedding=DeterministicFakeEmbedding(size=64))
store.add_documents(split_docs(load_docs()))
graph = build_graph(ScriptedLLM(), store.as_retriever())
