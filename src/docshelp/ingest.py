"""Load the product docs, split them into chunks and index them in a vector store.

This is the classic LangChain RAG ingestion pipeline:
    load (Documents) -> split (smaller Documents) -> embed + store (VectorStore) -> retrieve (Retriever)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from docshelp.config import DOCS_DIR, get_embeddings


def load_docs(docs_dir: Path = DOCS_DIR) -> list[Document]:
    return [
        Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name})
        for path in sorted(docs_dir.glob("*.md"))
    ]


def split_docs(docs: list[Document]) -> list[Document]:
    """Split on Markdown headings first (keeps sections intact), then cap chunk size."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section")], strip_headers=False
    )
    size_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    chunks: list[Document] = []
    for doc in docs:
        for section in header_splitter.split_text(doc.page_content):
            section.metadata = {**doc.metadata, **section.metadata}
            chunks.extend(size_splitter.split_documents([section]))
    return chunks


def build_vector_store(embeddings: Embeddings | None = None) -> InMemoryVectorStore:
    store = InMemoryVectorStore(embedding=embeddings or get_embeddings())
    store.add_documents(split_docs(load_docs()))
    return store


@lru_cache(maxsize=1)
def get_retriever(k: int = 4) -> BaseRetriever:
    """A Retriever is a Runnable: `retriever.invoke("question") -> list[Document]`."""
    return build_vector_store().as_retriever(search_kwargs={"k": k})


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[source: {d.metadata.get('source', '?')}]\n{d.page_content}" for d in docs
    )
