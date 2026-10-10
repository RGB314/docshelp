"""Memory schema as a collection: a growing list of past issues, each updated or inserted.

Academy: Intro to LangGraph M5 L4 "Memory Schema + Collection".

Concepts:
  * a collection stores many small items (one `PastIssue` each) instead of one big document. Each item
    is its own Store entry under ("issues", account_id), so it can be searched (lesson 4.6) and updated
    on its own.
  * `create_extractor(..., enable_inserts=True)` gets the existing items with their keys. The model can
    patch an existing item (the result's metadata has its `json_doc_id`) or insert a new one.
  * profile vs collection: a profile fits facts with one true value (OS, role); a collection fits
    things that accumulate (issues, conversations, preferences learned over time)

Run:  uv run poe lesson 4.8
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.store.memory import InMemoryStore
from trustcall import create_extractor

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.memory import PastIssue, ns, update_issues

ACCOUNT = "acct-1001"
MESSAGES = [
    "Notes I create on my iPhone take 20 minutes to appear on my Windows laptop.",
    "Good news: updating to 4.12 fixed the iPhone delay. New problem though: PDF export drops my images.",
]


def run(llm: BaseChatModel, store: InMemoryStore | None = None) -> list[list[dict]]:
    store = store or InMemoryStore()
    extractor = create_extractor(llm, tools=[PastIssue], tool_choice="PastIssue", enable_inserts=True)
    snapshots = []
    for text in MESSAGES:
        header(f"Customer: {text}")
        conversation = [SystemMessage("Record the customer's issues and their outcomes."), HumanMessage(text)]
        keys = update_issues(store, ACCOUNT, extractor, conversation)
        print(f"  wrote items: {keys}")
        items = sorted(store.search(ns("issues", ACCOUNT)), key=lambda item: item.key)
        snapshots.append([{"key": item.key, **item.value} for item in items])
        for item in snapshots[-1]:
            print(f"  {item['key']}: {item['summary']} [{item['status']}] {item.get('resolution') or ''}")
    return snapshots


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm())


if __name__ == "__main__":
    main()
