"""RAG: answer from your own documents (load -> split -> embed -> store -> retrieve -> generate).

Academy: not a separate Academy lesson, but the foundation for "Tools" (LangChain Essentials L4) and every
agent after it: the docs-search tool in lesson 1.6 is this retriever.

Concepts:
  * Documents with metadata (the source file name, used for citations)
  * splitting by Markdown headings, then by size
  * local embeddings (fastembed, free) and a vector store; a Retriever is a Runnable
  * a RAG chain in LCEL: {"context": retriever | format_docs, "question": passthrough} | prompt | llm

Run:  uv run poe lesson 1.5
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.ingest import format_docs, get_retriever, load_docs, split_docs

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer using only this Nimbus Notes documentation. Cite sources in [brackets].\n\n{context}"),
        ("human", "{question}"),
    ]
)


def run(llm: BaseChatModel, retriever: BaseRetriever) -> dict:
    header("1. Ingest: load and split the docs")
    docs = load_docs()
    chunks = split_docs(docs)
    print(f"Loaded {len(docs)} files, split into {len(chunks)} chunks. First chunk's metadata:")
    print(f"  {chunks[0].metadata}")

    header("2. Semantic search")
    question = "How long do I have to get my money back?"
    for d in retriever.invoke(question):
        print(f"- {d.metadata['source']:32} {d.page_content[:70].replace(chr(10), ' ')}...")
    print("(No keyword overlap with 'refund', but embeddings still find the right section.)")

    header("3. RAG chain: retriever | prompt | model")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    answer = rag_chain.invoke(question)
    print(f"Q: {question}\nA: {answer}")

    return {"files": len(docs), "chunks": len(chunks), "answer": answer}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    print("Embedding locally with fastembed (first run downloads a ~70 MB model)...")
    run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
