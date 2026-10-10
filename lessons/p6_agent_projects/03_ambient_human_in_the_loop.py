"""Human in the loop for an ambient agent: the Agent Inbox protocol.

Academy: Project: Ambient Agents M4 "Human-in-the-Loop".

Concepts (`build_inbox_agent(..., hitl=True)`):
  * an ambient agent has no chat window, so it asks for help the way a colleague would: it pauses
    and leaves a request in an inbox. The open-source Agent Inbox UI (Part 8 connects it to the server)
    reads interrupts in this shape:
        {"action_request": {"action", "args"}, "config": {allow_accept, allow_edit, allow_respond,
         allow_ignore}, "description": "Markdown for the reviewer"}
    and resumes with `Command(resume=[{"type": "accept" | "edit" | "respond"... , "args": ...}])`
  * which actions pause: sending a reply and opening a ticket (accept / edit / ignore / respond), and
    `ask_human` (respond / ignore); "notify" emails pause after triage to ask whether to answer
  * "respond" feedback isn't executed: it goes back to the model as a tool result, and the model drafts
    again. "ignore" ends the run.
  * Part 2.7 did approvals with `HumanInTheLoopMiddleware`, and Part 3.13 with raw `interrupt()`. This is
    the protocol a UI and a team of reviewers use.

Run:  uv run poe lesson 6.3                (scripted support lead)
      uv run poe lesson 6.3 --interactive  (you are the support lead)
"""

import argparse
import json

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from docshelp.ambient import Reviewer, build_inbox_agent, load_inbox, process_inbox
from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.ingest import get_retriever

EMAILS = ["msg-005", "msg-001", "msg-003"]  # a billing alert, a known issue, an urgent outage


def scripted_lead(email_id: str, request: dict) -> dict:
    """What a support lead might do with each request (a stand-in for a person in Agent Inbox)."""
    action, args = request["action_request"]["action"], request["action_request"]["args"]
    if action.startswith("Email needs your attention"):
        return {"type": "ignore", "args": None}  # billing alerts: seen, nothing to reply
    if action == "open_ticket":
        return {"type": "edit", "args": {"action": action, "args": {**args, "priority": "urgent"}}}
    if action == "send_reply" and email_id == "msg-001" and "status" not in args.get("body", "").lower():
        return {"type": "response", "args": "Also link the status page: status.nimbusnotes.example"}
    return {"type": "accept", "args": None}


def terminal_lead(email_id: str, request: dict) -> dict:
    config = request["config"]
    allowed = [k[6:] for k, v in config.items() if v]  # allow_accept -> accept, ...
    print(f"  {request['action_request']['action']}: {json.dumps(request['action_request']['args'], indent=2)}")
    choice = input(f"  {'/'.join(allowed)} > ").strip().lower() or allowed[0]
    if choice == "edit":
        body = input("  new body: ").strip()
        return {"type": "edit", "args": {"action": request["action_request"]["action"],
                                         "args": {**request["action_request"]["args"], "body": body}}}
    if choice == "respond":
        return {"type": "response", "args": input("  feedback: ").strip()}
    return {"type": choice, "args": None}


def run(
    llm: BaseChatModel,
    retriever: BaseRetriever,
    lead: Reviewer = scripted_lead,
    email_ids: list[str] | None = None,
) -> dict:
    agent = build_inbox_agent(llm, retriever, hitl=True)
    emails = [e for e in load_inbox() if e["id"] in (email_ids or EMAILS)]
    header(f"Processing {len(emails)} emails; each pause goes to the support lead")
    # process_inbox plays the Agent Inbox UI: it shows each request to the lead and resumes with
    # Command(resume=[response]). See src/docshelp/ambient.py.
    return process_inbox(agent, emails, lead)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true", help="answer the Agent Inbox requests yourself")
    args = parser.parse_args()
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever(), terminal_lead if args.interactive else scripted_lead)


if __name__ == "__main__":
    main()
