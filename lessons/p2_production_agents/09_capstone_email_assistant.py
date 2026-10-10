"""Capstone: a support email assistant with verification, dynamic tools and human approval.

Academy: Intro to LangChain M3 L5 project ("Email Assistant"), applied to Nimbus Notes support email.

Brings Part 2 together in `src/docshelp/email_assistant.py::build_email_assistant`:
  * custom state: `verify_sender` records the verified account with `Command(update=...)` (lesson 2.2)
  * dynamic tools: `lookup_account` is invisible until verification succeeds (lesson 2.8)
  * defence in depth: even if called, `lookup_account` reads the account from state, not from arguments
  * human in the loop: every `send_reply` waits for approve / edit / reject (lesson 2.7)
  * summarization for long threads, a model-call limit, and the `AuditLog` middleware writing
    data/audit.jsonl (2.4–2.6)

Run:  uv run poe lesson 2.9              (scripted reviewer approves the reply)
      uv run poe lesson 2.9 --review     (you review the drafted reply)
"""

import argparse
import uuid
from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.retrievers import BaseRetriever
from langgraph.types import Command

from docshelp.config import DATA_DIR, describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.email_assistant import build_email_assistant
from docshelp.ingest import get_retriever
from docshelp.middleware import AuditLog

EMAIL = {
    "from": "ada@acme.example",
    "subject": "Adding seats",
    "body": (
        "Hi! We're on account acct-1001 and nearly out of seats. How do I add 3 more people, "
        "and what would our Team plan cost per month with 15 members? Thanks, Ada"
    ),
}


def approve(action: dict) -> dict:
    return {"type": "approve"}


def terminal_review(action: dict) -> dict:
    choice = input("  [a]pprove, [e]dit body, [r]eject? ").strip().lower()[:1]
    if choice == "e":
        body = input("  new body: ").strip()
        return {"type": "edit", "edited_action": {"name": action["name"], "args": {**action["args"], "body": body}}}
    if choice == "r":
        return {"type": "reject", "message": input("  what should change? ").strip() or "Please rewrite it."}
    return {"type": "approve"}


def run(
    llm: BaseChatModel,
    retriever: BaseRetriever,
    email: dict | None = None,
    review: Callable[[dict], dict] = approve,
    *,
    summary_llm: BaseChatModel | None = None,
    audit: AuditLog | None = None,
) -> dict:
    email = email or EMAIL
    agent = build_email_assistant(llm, retriever, summary_llm=summary_llm, audit=audit)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    header(f"Email from {email['from']}: {email['subject']}")
    print(email["body"])
    task = f"New customer email.\nFrom: {email['from']}\nSubject: {email['subject']}\n\n{email['body']}"
    result = agent.invoke({"messages": [{"role": "user", "content": task}]}, config)
    interrupts = []
    while "__interrupt__" in result:
        decisions = []
        for action in result["__interrupt__"][0].value["action_requests"]:
            args = action["args"]
            header(f"Review: send_reply to {args.get('to')} / {args.get('subject')}")
            print(args.get("body", ""))
            interrupts.append(action)
            decisions.append(review(action))
        result = agent.invoke(Command(resume={"decisions": decisions}), config)
    header("What happened")
    for message in result["messages"]:
        if isinstance(message, ToolMessage):
            print(f"  [{message.name}] {message.text[:160]}")
    print(f"  verified account: {result.get('verified_account') or '(none)'}")
    answer = result["messages"][-1].text
    print(f"Assistant: {answer}")
    return {"state": result, "interrupts": interrupts, "answer": answer}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", action="store_true", help="review the drafted reply yourself")
    args = parser.parse_args()
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(
        get_llm(),
        get_retriever(),
        review=terminal_review if args.review else approve,
        summary_llm=get_llm(fast=True),
        audit=AuditLog(DATA_DIR / "audit.jsonl"),
    )


if __name__ == "__main__":
    main()
