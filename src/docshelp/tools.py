"""Tools the model can call.

`@tool` turns a typed, documented Python function into a LangChain tool: the function name,
docstring and type hints become the schema the model sees.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Literal

from langchain.tools import ToolRuntime
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, tool

from docshelp.config import DATA_DIR
from docshelp.customers import ACCOUNTS, Customer
from docshelp.ingest import format_docs, get_retriever

TICKETS_PATH = DATA_DIR / "tickets.jsonl"
OUTBOX_PATH = DATA_DIR / "outbox.jsonl"  # where "sent" emails go; nothing is really emailed


def make_search_docs(retriever: BaseRetriever | None = None) -> BaseTool:
    """Build the `search_docs` tool over a given retriever (tests pass one with fake embeddings)."""

    @tool
    def search_docs(query: str) -> str:
        """Search the Nimbus Notes product documentation. Use this for any question about the product."""
        return format_docs((retriever or get_retriever()).invoke(query))

    return search_docs


search_docs = make_search_docs()  # uses the real retriever, built on first call


@tool
def quote_price(
    plan: Literal["free", "team", "enterprise"],
    members: int,
    billing: Literal["monthly", "annual"] = "monthly",
) -> str:
    """Get an exact price quote from the billing system for a plan, number of members and billing cycle."""
    if plan == "free":
        return "The Free plan costs $0."
    if plan == "enterprise":
        return "Enterprise pricing is custom. Please contact sales@nimbusnotes.example."
    monthly = 8 * members
    if billing == "annual":
        yearly = monthly * 12 * 0.8
        return f"Team plan, {members} members, annual billing: ${yearly:,.2f} per year (20% discount applied)."
    return f"Team plan, {members} members, monthly billing: ${monthly:,.2f} per month."


def _append(path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def save_ticket(ticket: dict) -> str:
    """Pretend to file a support ticket (appends to data/tickets.jsonl) and return its id."""
    ticket_id = f"NIM-{uuid.uuid4().hex[:6].upper()}"
    _append(TICKETS_PATH, {"id": ticket_id, "created_at": datetime.now(timezone.utc).isoformat(), **ticket})
    return ticket_id


# --- Tools that use runtime context (Part 2) ---------------------------------------------------------
# A `runtime: ToolRuntime` parameter is injected by the agent and hidden from the model: the model sees
# only the other arguments. `runtime.context` is the run's `Customer`, so these tools act on the caller's
# own account and the model can't ask for anyone else's.


@tool
def my_account(runtime: ToolRuntime[Customer]) -> str:
    """Look up the customer's own account: plan, seats used, renewal date and data region."""
    customer = runtime.context or Customer()
    return ACCOUNTS[customer.account_id].describe()


@tool
def open_ticket(
    summary: str,
    priority: Literal["low", "normal", "high", "urgent"],
    runtime: ToolRuntime[Customer],
) -> str:
    """Open a support ticket so a human engineer follows up. Use for bugs, data loss or account problems."""
    customer = runtime.context or Customer()
    ticket_id = save_ticket(
        {"summary": summary, "priority": priority, "customer": customer.name, "account_id": customer.account_id}
    )
    return f"Opened ticket {ticket_id} ({priority} priority)."


@tool
def send_reply(to: str, subject: str, body: str) -> str:
    """Send an email reply to a customer. The reply goes out exactly as written."""
    sent_at = datetime.now(timezone.utc).isoformat()
    _append(OUTBOX_PATH, {"to": to, "subject": subject, "body": body, "sent_at": sent_at})
    return f"Sent '{subject}' to {to}."
