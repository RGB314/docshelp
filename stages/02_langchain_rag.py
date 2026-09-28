"""Stage 2 - LangChain RAG and tool-calling agents.

What LangChain gives you here:
  * document loaders, text splitters, embeddings and vector stores behind common interfaces
  * a Retriever is just another Runnable, so it drops straight into an LCEL chain
  * `@tool` + `create_agent`: the model decides which tools to call and in what order

Run:  uv run poe stage2        (Docker: docker compose run --rm app poe stage2)
"""

from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.ingest import format_docs, get_retriever, load_docs, split_docs
from docshelp.tools import quote_price, search_docs


def header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    ensure_llm_ready()
    llm = get_llm()
    print(f"Using {describe()}")

    # 1. Ingestion: load -> split -> embed -> store.
    header("1. Ingest the docs")
    docs = load_docs()
    chunks = split_docs(docs)
    print(f"Loaded {len(docs)} files, split into {len(chunks)} chunks.")
    print("Embedding locally with fastembed (first run downloads a ~70 MB model)...")
    retriever = get_retriever()

    # 2. Retrieval on its own.
    header("2. Semantic search")
    question = "How long do I have to get my money back?"
    for d in retriever.invoke(question):
        print(f"- {d.metadata['source']:32} {d.page_content[:70].replace(chr(10), ' ')}...")
    print("(No keyword overlap with 'refund', but embeddings still find the right section.)")

    # 3. A RAG chain in LCEL. The dict runs both branches in parallel on the same input.
    header("3. RAG chain: retriever | prompt | model")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer using only this Nimbus Notes documentation. Cite sources in [brackets].\n\n{context}",
            ),
            ("human", "{question}"),
        ]
    )
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    for q in [question, "Can I use Nimbus offline in the browser?"]:
        print(f"\nQ: {q}\nA: {rag_chain.invoke(q)}")

    # 4. A tool-calling agent. Unlike the fixed chain above, the model chooses the steps:
    #    here it should search the docs AND call the pricing tool.
    header("4. Tool-calling agent (create_agent)")
    agent = create_agent(
        llm,
        tools=[search_docs, quote_price],
        system_prompt="You are DocsHelp, the Nimbus Notes support assistant. Use tools; don't guess.",
    )
    q = (
        "We're a team of 12 and need SSO with Okta. Which plan do we need, and what would the "
        "Team plan cost us per year with annual billing?"
    )
    print(f"Q: {q}\n")
    result = agent.invoke({"messages": [{"role": "user", "content": q}]})
    for msg in result["messages"][1:]:
        for call in getattr(msg, "tool_calls", []) or []:
            print(f"  -> tool call: {call['name']}({call['args']})")
    print(f"\nA: {result['messages'][-1].text}")

    print(
        "\n`create_agent` returns a compiled LangGraph graph. Stage 3 builds our own graph "
        "to get control flow the generic agent loop doesn't have."
    )


if __name__ == "__main__":
    main()
