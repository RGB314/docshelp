"""An ambient agent: DocsHelp works through the support inbox on its own.

Academy: Project: Ambient Agents M2 "Building Agents" (the email assistant: triage router + response
agent), applied to the Nimbus Notes support inbox in `data/inbox/`.

Concepts (the graph is `src/docshelp/ambient.py::build_inbox_agent`):
  * ambient vs chat: the agent is triggered by *events* (a new email), runs in the background and only
    involves a human when it must. Each email is one run on its own thread.
  * a triage router first: respond / notify / ignore, with structured output (lessons 1.3, 3.2).
    Cheap, and it keeps the response agent away from newsletters and auto-replies.
  * a response agent with tools, built by hand (lesson 3.3): research tools (docs, changelog, prices),
    action tools (`open_ticket`, `send_reply`) and a `done` tool so the agent says explicitly when the
    email is handled
  * the sender's account is looked up from the email address, so tickets land on the right account

Run:  uv run poe lesson 6.1
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.retrievers import BaseRetriever

from docshelp.ambient import build_inbox_agent, load_inbox, public, tools_called
from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.ingest import get_retriever


def run(llm: BaseChatModel, retriever: BaseRetriever, email_ids: list[str] | None = None) -> dict:
    agent = build_inbox_agent(llm, retriever)
    emails = [e for e in load_inbox() if email_ids is None or e["id"] in email_ids]
    results = {}
    for email in emails:  # in production: a webhook or a cron job per new email (Part 8)
        header(f"{email['id']}  {email['from']}: {email['subject']}")
        state = agent.invoke({"email": public(email)})
        calls = tools_called(state.get("messages", []))
        sent = [m.text for m in state.get("messages", []) if isinstance(m, ToolMessage) and m.name == "send_reply"]
        reply = sent[-1] if sent else None
        results[email["id"]] = {"classification": state["classification"], "tools": calls, "reply": reply}
        print(f"  triage: {state['classification']}")
        if calls:
            print(f"  tools:  {', '.join(calls)}")
        if reply:
            print(f"  {reply}")
    header("Inbox summary")
    for label in ("respond", "notify", "ignore"):
        ids = [i for i, r in results.items() if r["classification"] == label]
        print(f"  {label:8} {len(ids)}  {', '.join(ids)}")
    return results


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
