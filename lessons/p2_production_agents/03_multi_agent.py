"""Multi-agent systems: a supervisor delegating to docs, billing and status specialists.

Academy: Intro to LangChain M2 L3 "Multi-Agent Systems" and M2 L4 project ("Wedding Planner": a
coordinator with specialist agents, one of them backed by an MCP server), applied to a support team.

Concepts:
  * subagents as tools: each specialist is a `create_agent` agent wrapped in an `@tool`
    (`src/docshelp/team.py`). The supervisor only sees questions in and answers out.
  * context isolation: specialists start with a fresh message list, so the supervisor must ask complete,
    standalone questions, and its own context stays small
  * passing runtime context down: the billing specialist reads the caller's account via `runtime.context`
  * mixing tool sources: the status specialist's tools come from the MCP server of lesson 2.1
  * cost: specialists can run on the fast model while the supervisor reasons with the default one

Run:  uv run poe lesson 2.3
"""

import asyncio
from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.customers import Customer
from docshelp.ingest import get_retriever
from docshelp.mcp_server import load_ops_tools
from docshelp.team import build_support_team

QUESTION = (
    "Sync has been slow all morning for our team in eu-west. Is that on your side? "
    "Also, can I invite one more colleague without changing plan?"
)


async def ask(supervisor, customer: Customer) -> dict:
    result = await supervisor.ainvoke({"messages": [{"role": "user", "content": QUESTION}]}, context=customer)
    consulted = []
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                consulted.append(call["name"])
                print(f"  supervisor -> {call['name']}: {call['args'].get('question', '')}")
        elif isinstance(message, ToolMessage):
            print(f"  {message.name} <- {message.text}")
    return {"messages": result["messages"], "consulted": consulted, "answer": result["messages"][-1].text}


def run(
    llm: BaseChatModel,
    retriever: BaseRetriever,
    status_tools: Sequence[BaseTool],
    *,
    specialist_llm: BaseChatModel | None = None,
    customer: Customer | None = None,
) -> dict:
    customer = customer or Customer(name="Ada", plan="team", account_id="acct-1001")
    supervisor = build_support_team(llm, retriever, status_tools, specialist_llm=specialist_llm)
    header(f"{customer.name} asks the support team")
    print(f"User: {QUESTION}")
    result = asyncio.run(ask(supervisor, customer))
    print(f"DocsHelp: {result['answer']}")
    return result


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    status_tools = asyncio.run(load_ops_tools())  # the MCP server from lesson 2.1
    run(get_llm(), get_retriever(), status_tools, specialist_llm=get_llm(fast=True))


if __name__ == "__main__":
    main()
