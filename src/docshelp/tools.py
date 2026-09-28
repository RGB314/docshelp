"""Tools the model can call.

`@tool` turns a typed, documented Python function into a LangChain tool: the function name,
docstring and type hints become the schema the model sees.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Literal

from langchain_core.tools import tool

from docshelp.config import DATA_DIR
from docshelp.ingest import format_docs, get_retriever

TICKETS_PATH = DATA_DIR / "tickets.jsonl"


@tool
def search_docs(query: str) -> str:
    """Search the Nimbus Notes product documentation. Use this for any question about the product."""
    return format_docs(get_retriever().invoke(query))


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


def save_ticket(ticket: dict) -> str:
    """Pretend to file a support ticket (appends to data/tickets.jsonl) and return its id."""
    ticket_id = f"NIM-{uuid.uuid4().hex[:6].upper()}"
    record = {"id": ticket_id, "created_at": datetime.now(timezone.utc).isoformat(), **ticket}
    TICKETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TICKETS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return ticket_id
