"""Semantic search over memories: find past issues by meaning, not by key.

Academy: Intro to LangGraph M5 L2 "LangGraph Store" (search and indexing).

Concepts:
  * a store created with an `index` (an embeddings model, its dimensions, and which `fields` to embed)
    embeds every item you `put`; `store.search(namespace, query="...")` then ranks items by meaning,
    with a `score` per hit. Local fastembed embeddings keep this free (lesson 1.5).
  * `filter={"status": "open"}` narrows by exact field values; `limit` caps the results
  * `put(..., index=False)` stores an item without embedding it (for data you'll never search)
  * production: `PostgresStore` with pgvector offers the same interface; LangGraph Server configures
    the index in `langgraph.json` (Part 8)
  * use: before answering, pull the customer's most relevant past issues into the prompt

Run:  uv run poe lesson 4.6
"""

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from docshelp.config import describe, ensure_llm_ready, get_embeddings, get_llm
from docshelp.console import header
from docshelp.memory import make_store, ns
from docshelp.sources import load_sample_tickets

QUESTION = "My notes from the phone take ages to show up on my laptop again."


def run(llm: BaseChatModel, embeddings: Embeddings) -> dict:
    store = make_store(embeddings, fields=["summary"])  # embed only the summary field
    for ticket in load_sample_tickets():
        store.put(ns("issues", ticket["account_id"]), ticket["id"], ticket)
    store.put(ns("issues", "acct-1001"), "note", {"summary": "internal note", "status": "solved"}, index=False)

    account = "acct-1001"
    header(f"Search {account}'s past issues for: {QUESTION!r}")
    hits = store.search(ns("issues", account), query=QUESTION, limit=2)
    for hit in hits:
        print(f"  {hit.score:.2f}  {hit.value['summary']} ({hit.value['status']})")

    header("Only open issues (filter), across all of this account's items")
    open_hits = store.search(ns("issues", account), query=QUESTION, filter={"status": "open"}, limit=5)
    for hit in open_hits:
        print(f"  {hit.score:.2f}  {hit.value['summary']}")

    header("Answer with the relevant memories in the prompt")
    memories = "\n".join(f"- {h.value['summary']} ({h.value['status']})" for h in hits)
    prompt = (
        "You are DocsHelp. The customer has reported these related issues before:\n"
        f"{memories}\n\nAcknowledge the history briefly, then help with: {QUESTION}"
    )
    answer = llm.invoke(prompt).text
    print(f"DocsHelp: {answer}")
    return {"hits": [h.value for h in hits], "open_hits": [h.value for h in open_hits], "answer": answer}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_embeddings())


if __name__ == "__main__":
    main()
