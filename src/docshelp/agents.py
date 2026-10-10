"""The DocsHelp support agent built with LangChain's `create_agent` (Part 1 capstone).

`create_agent` is LangChain's prebuilt agent loop: the model calls tools until it can answer. Later
parts extend this same agent: middleware (Part 2), evaluation (Part 5) and deployment (Part 8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langgraph.types import Checkpointer
from pydantic import BaseModel, Field

from docshelp.tools import make_search_docs, quote_price

Plan = Literal["free", "team", "enterprise"]


@dataclass
class Customer:
    """Runtime context: facts about the caller that the app knows (not the model). Passed per run."""

    name: str = "there"
    plan: Plan = "free"


class SupportAnswer(BaseModel):
    """The agent's final reply, as structured data the app can act on."""

    answer: str = Field(description="The reply to show the customer.")
    sources: list[str] = Field(default_factory=list, description="Doc file names the answer relies on.")
    needs_ticket: bool = Field(description="True if a human should follow up via a support ticket.")


PLAN_GUIDANCE = {
    "free": "They are on the Free plan. If they need a paid feature, say which plan includes it.",
    "team": "They are on the Team plan. Mention Team-plan limits when relevant.",
    "enterprise": "They are on the Enterprise plan with a dedicated account manager; offer that contact.",
}


def support_prompt(customer: Customer) -> str:
    return (
        "You are DocsHelp, the support assistant for Nimbus Notes. "
        f"You are talking to {customer.name}. {PLAN_GUIDANCE[customer.plan]} "
        "Always search the docs before answering product questions, and quote prices only from the "
        "quote_price tool. Never guess. Cite doc file names as sources."
    )


@dynamic_prompt
def plan_aware_prompt(request: ModelRequest) -> str:
    """Middleware that rebuilds the system prompt on every model call from the run's context."""
    return support_prompt(request.runtime.context or Customer())


def build_support_agent(
    llm: BaseChatModel,
    retriever: BaseRetriever | None = None,
    checkpointer: Checkpointer | None = None,
    *,
    structured: bool = True,
):
    """A tool-calling support agent with memory (checkpointer) and plan-aware prompting.

    With `structured=True` the final reply is a `SupportAnswer` in `result["structured_response"]`.
    """
    return create_agent(
        llm,
        tools=[make_search_docs(retriever), quote_price],
        middleware=[plan_aware_prompt],
        context_schema=Customer,
        response_format=SupportAnswer if structured else None,
        checkpointer=checkpointer,
        name="docshelp_agent",
    )
