"""The support email assistant: Part 2's capstone, combining everything from Parts 1 and 2.

A support rep forwards a customer email. The agent:
  1. verifies the sender against the account they name (`verify_sender` writes custom state),
  2. only then gets the account tools (dynamic tools: a `wrap_model_call` reads that state),
  3. researches the answer (RAG over the docs, prices from the billing tool),
  4. proposes a reply, which a human approves, edits or rejects before it is sent (HITL),
while summarization keeps long threads small, a call limit stops runaway loops, and an optional `AuditLog`
records every step.
"""

from __future__ import annotations

from typing import NotRequired

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ModelRequest,
    SummarizationMiddleware,
    wrap_model_call,
)
from langchain.tools import ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Checkpointer, Command

from docshelp.customers import ACCOUNTS
from docshelp.middleware import AuditLog
from docshelp.tools import make_search_docs, quote_price, send_reply

PROMPT = (
    "You are the Nimbus Notes support email assistant. For each customer email: first call verify_sender "
    "with the sender's address and the account id they mention. Only share account details if verification "
    "succeeded. Use search_docs for how-to answers and quote_price for prices; never guess. Finally send one "
    "concise, friendly reply with send_reply, addressed to the sender."
)
ACCOUNT_TOOLS = {"lookup_account"}  # hidden from the model until the sender is verified


class EmailState(AgentState):
    """Agent state plus the account the sender proved they belong to."""

    verified_account: NotRequired[str | None]


@tool
def verify_sender(email: str, account_id: str, runtime: ToolRuntime) -> Command:
    """Check that the email's sender is a registered contact of the account they name. Do this first."""
    account = ACCOUNTS.get(account_id.strip())
    verified = account is not None and email.strip().lower() in account.contacts
    text = (
        f"Verified: {email} is a contact of {account_id}."
        if verified
        else f"Not verified: {email} is not a contact of {account_id}. Do not share any account details."
    )
    update: dict = {"messages": [ToolMessage(text, tool_call_id=runtime.tool_call_id)]}
    if verified:
        update["verified_account"] = account_id.strip()
    return Command(update=update)


@tool
def lookup_account(runtime: ToolRuntime) -> str:
    """Plan, seats, renewal date and data region of the verified sender's account."""
    account_id = runtime.state.get("verified_account")  # from state, never from the model's arguments
    return ACCOUNTS[account_id].describe() if account_id else "Refused: verify the sender first."


@wrap_model_call
def unlock_after_verification(request: ModelRequest, handler):
    """Dynamic tools from state: account tools appear only once `verified_account` is set."""
    if not request.state.get("verified_account"):
        request = request.override(tools=[t for t in request.tools if getattr(t, "name", None) not in ACCOUNT_TOOLS])
    return handler(request)


def build_email_assistant(
    llm: BaseChatModel,
    retriever: BaseRetriever | None = None,
    checkpointer: Checkpointer | None = None,
    *,
    summary_llm: BaseChatModel | None = None,
    audit: AuditLog | None = None,
):
    middleware = [
        ModelCallLimitMiddleware(run_limit=12, exit_behavior="end"),  # a looping model can't run up a bill
        unlock_after_verification,
        SummarizationMiddleware(model=summary_llm or llm, trigger=("tokens", 6000), keep=("messages", 20)),
        HumanInTheLoopMiddleware(interrupt_on={"send_reply": {"allowed_decisions": ["approve", "edit", "reject"]}}),
    ]
    if audit:
        middleware.append(audit)
    return create_agent(
        llm,
        tools=[verify_sender, lookup_account, make_search_docs(retriever), quote_price, send_reply],
        system_prompt=PROMPT,
        middleware=middleware,
        state_schema=EmailState,
        checkpointer=checkpointer or InMemorySaver(),  # HITL pauses need a checkpointer
        name="email_assistant",
    )
