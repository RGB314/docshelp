"""Memory schema as a profile: one structured document per customer, patched as you learn more.

Academy: Intro to LangGraph M5 L3 "Memory Schema + Profile".

Concepts:
  * a schema (`CustomerProfile` in `src/docshelp/memory.py`) makes memory predictable: the app can read
    `profile["operating_systems"]` instead of parsing free text (lesson 4.5)
  * the hard part is *updating*: regenerating the whole profile risks dropping fields the latest message
    didn't mention. `trustcall.create_extractor` extracts the first version, then asks the model only for
    JSON patches against the existing document (`existing={"CustomerProfile": ...}`), and retries with
    the validation error if the output doesn't fit the schema
  * the profile lives in the Store under ("profile", account_id), key "profile": one item, overwritten

Run:  uv run poe lesson 4.7
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.store.memory import InMemoryStore
from trustcall import create_extractor

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.memory import CustomerProfile, ns, update_profile

ACCOUNT = "acct-1001"
MESSAGES = [
    "Hi, I'm Ada, the workspace admin at Acme. I use Nimbus on my Windows laptop and my iPhone.",
    "Quick update: I've switched to a MacBook. Also, please keep answers short.",
]


def run(llm: BaseChatModel, store: InMemoryStore | None = None) -> list[dict]:
    store = store or InMemoryStore()
    extractor = create_extractor(llm, tools=[CustomerProfile], tool_choice="CustomerProfile")
    profiles = []
    for text in MESSAGES:
        header(f"Customer: {text}")
        conversation = [SystemMessage("Extract facts about the customer from this message."), HumanMessage(text)]
        update_profile(store, ACCOUNT, extractor, conversation)
        saved = store.get(ns("profile", ACCOUNT), "profile")
        profiles.append(saved.value if saved else {})
        for field, value in profiles[-1].items():
            print(f"  {field:18} {value}")
    return profiles


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm())


if __name__ == "__main__":
    main()
