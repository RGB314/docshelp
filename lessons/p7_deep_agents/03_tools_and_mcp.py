"""Tools and MCP in a deep agent: your tools sit next to the harness's own.

Academy: Introduction to Deep Agents M1 "Tools", "MCP".

Concepts:
  * `tools=[...]` adds to the built-in tools (files, `task`); it never replaces them. Any LangChain tool
    works unchanged, including tools that read runtime context: `my_account` (lesson 2.2) still sees only
    the caller's account because `context_schema=Customer` and `context=...` pass straight through.
  * MCP tools are just more tools: load them with `langchain-mcp-adapters` (lesson 2.1) and pass the list.
    Here they come from the Nimbus Ops status server, started as a subprocess over stdio.
  * MCP tools are async, so the agent runs with `ainvoke`. (Sync tools work under `ainvoke` too.)
  * fewer, well-described tools beat many: each tool's name and docstring are in every model call.

Use case: a customer in eu-west asks why sync is slow. The agent checks their account's region, then the
live status for that region.

Run:  uv run poe lesson 7.3
"""

import asyncio

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.customers import Customer
from docshelp.deep import deep_agent
from docshelp.mcp_server import load_ops_tools
from docshelp.tools import my_account

QUESTION = "Sync has been slow all morning. Is something wrong on your side?"
CUSTOMER = Customer(name="Ada", plan="team", account_id="acct-1001")
PROMPT = "You are Nimbus Notes support. Check the customer's account region, then the live status, then answer briefly."


async def run_async(llm: BaseChatModel) -> dict:
    mcp_tools = await load_ops_tools()  # service_status and list_incidents, from the MCP server process
    agent = deep_agent(llm, tools=[my_account, *mcp_tools], system_prompt=PROMPT, context_schema=Customer)
    header(f"Tools from MCP: {', '.join(t.name for t in mcp_tools)}")
    result = await agent.ainvoke({"messages": [{"role": "user", "content": QUESTION}]}, context=CUSTOMER)
    tool_results = {m.name: m.text for m in result["messages"] if isinstance(m, ToolMessage)}
    for name, text in tool_results.items():
        print(f"  {name} -> {text[:120]}")
    header("Answer")
    print(result["messages"][-1].text)
    return {"tool_results": tool_results, "answer": result["messages"][-1].text}


def run(llm: BaseChatModel) -> dict:
    return asyncio.run(run_async(llm))


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm())


if __name__ == "__main__":
    main()
