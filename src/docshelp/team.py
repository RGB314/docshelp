"""A multi-agent support team: a supervisor that delegates to specialist agents (Part 2.3).

Each specialist is an ordinary `create_agent` agent with a narrow job, its own prompt and its own tools.
The supervisor sees each specialist as just another tool (`ask_docs`, `ask_billing`, `ask_status`), so it
decides who to consult and combines their answers. This "subagents as tools" pattern keeps every
agent's context small: the supervisor never sees the specialists' tool calls, only their final answers.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, tool
from langgraph.types import Checkpointer

from docshelp.customers import Customer
from docshelp.tools import make_search_docs, my_account, quote_price

SUPERVISOR_PROMPT = (
    "You lead the Nimbus Notes support team. Answer the customer by consulting your specialists: "
    "ask_docs for how-to questions, ask_billing for plans, prices, seats and their account, ask_status for "
    "outages and incidents. Ask each specialist a complete, standalone question; they can't see this chat. "
    "Then write one friendly reply that combines their answers."
)


def build_specialists(
    llm: BaseChatModel, retriever: BaseRetriever | None, status_tools: Sequence[BaseTool]
) -> dict[str, object]:
    return {
        "docs": create_agent(
            llm,
            tools=[make_search_docs(retriever)],
            system_prompt="You answer Nimbus Notes how-to questions. Always search the docs and cite file names.",
            name="docs_specialist",
        ),
        "billing": create_agent(
            llm,
            tools=[quote_price, my_account],
            system_prompt="You handle Nimbus Notes plans, prices, seats and the customer's own account.",
            context_schema=Customer,
            name="billing_specialist",
        ),
        "status": create_agent(
            llm,
            tools=list(status_tools),
            system_prompt="You report Nimbus Notes service status and incidents. Always check the status tools.",
            name="status_specialist",
        ),
    }


def build_support_team(
    llm: BaseChatModel,
    retriever: BaseRetriever | None = None,
    status_tools: Sequence[BaseTool] = (),
    *,
    specialist_llm: BaseChatModel | None = None,
    checkpointer: Checkpointer | None = None,
):
    """The supervisor agent. Specialists can run on a cheaper model (`specialist_llm`)."""
    team = build_specialists(specialist_llm or llm, retriever, status_tools)

    async def consult(name: str, question: str, runtime: ToolRuntime[Customer]) -> str:
        # Pass the caller's context down: the billing specialist needs to know whose account to read.
        request = {"messages": [{"role": "user", "content": question}]}
        result = await team[name].ainvoke(request, context=runtime.context)
        return result["messages"][-1].text

    @tool
    async def ask_docs(question: str, runtime: ToolRuntime[Customer]) -> str:
        """Ask the documentation specialist a how-to question about Nimbus Notes features."""
        return await consult("docs", question, runtime)

    @tool
    async def ask_billing(question: str, runtime: ToolRuntime[Customer]) -> str:
        """Ask the billing specialist about plans, prices, seats or the customer's own account."""
        return await consult("billing", question, runtime)

    @tool
    async def ask_status(question: str, runtime: ToolRuntime[Customer]) -> str:
        """Ask the status specialist whether there is an outage or incident, e.g. in a region."""
        return await consult("status", question, runtime)

    return create_agent(
        llm,
        tools=[ask_docs, ask_billing, ask_status],
        system_prompt=SUPERVISOR_PROMPT,
        context_schema=Customer,
        checkpointer=checkpointer,
        name="support_supervisor",
    )
