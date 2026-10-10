"""Memory for the ambient agent: learn the support lead's preferences from their corrections.

Academy: Project: Ambient Agents M5 "Memory".

Concepts (`build_inbox_agent(..., hitl=True, memory=True)`):
  * every human correction is a signal. An *edit* to a reply shows the preferred style. *Ignoring* a
    drafted reply means the email didn't need one. *Answering* a "notify" email means it did.
  * the agent keeps two preference documents in the Store (namespace `("inbox_preferences",)`): triage
    preferences and response preferences. They start from defaults and go into the prompts.
  * after a correction an LLM rewrites the affected document, changing only what the correction implies
    (structured output with its reasoning). This is profile-style memory (lesson 4.7), updated from
    feedback instead of from conversation.
  * the effect: later emails are triaged and answered the way the lead wants, without the lead repeating it.
    Part 9 automates this further, from LangSmith feedback.

Run:  uv run poe lesson 6.4
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langgraph.store.memory import InMemoryStore

from docshelp.ambient import DEFAULT_PREFERENCES, build_inbox_agent, get_preferences, load_inbox, process_inbox
from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.ingest import get_retriever

SIGN_OFF = "\n\nCheers,\nThe Nimbus Notes team"
EMAILS = ["msg-001", "msg-002", "msg-004"]  # a customer, a newsletter, another customer


def preferring_lead(email_id: str, request: dict) -> dict:
    """A support lead with habits: newsletters get no reply, and every reply ends with the team sign-off."""
    action, args = request["action_request"]["action"], request["action_request"]["args"]
    if action.startswith("Email needs your attention"):
        return {"type": "ignore", "args": None}
    if action == "send_reply" and email_id == "msg-002":
        return {"type": "ignore", "args": None}
    if action == "send_reply" and "Cheers" not in args.get("body", ""):
        return {"type": "edit", "args": {"action": action, "args": {**args, "body": args.get("body", "") + SIGN_OFF}}}
    return {"type": "accept", "args": None}


def run(llm: BaseChatModel, retriever: BaseRetriever, store: InMemoryStore | None = None) -> dict:
    store = store or InMemoryStore()  # LangGraph Server provides a persistent one (Part 8)
    header("Preferences before")
    for kind in ("triage", "response"):
        print(f"  {kind}: {get_preferences(store, kind)}")

    agent = build_inbox_agent(llm, retriever, hitl=True, memory=True, store=store)
    header("Processing the inbox with a support lead who corrects the agent")
    results = process_inbox(agent, [e for e in load_inbox() if e["id"] in EMAILS], preferring_lead)

    header("Preferences after (learned from the corrections)")
    learned = {kind: get_preferences(store, kind) for kind in ("triage", "response")}
    for kind, text in learned.items():
        changed = "changed" if text != DEFAULT_PREFERENCES[kind] else "unchanged"
        print(f"  {kind} ({changed}): {text}")
    return {"results": results, "preferences": learned}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
