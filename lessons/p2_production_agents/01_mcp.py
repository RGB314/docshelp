"""MCP: give the agent tools, data and prompts from a separate server process.

Academy: Intro to LangChain M2 L1 "Model Context Protocol"; LangChain Essentials L5 "Tools with MCP".

Concepts:
  * MCP standardises how apps get tools and context. One server (`src/docshelp/mcp_server.py`, the
    Nimbus Ops status page) can serve any MCP client; one client can use many servers.
  * `MultiServerMCPClient` (from `langchain-mcp-adapters`) starts the server over stdio and converts its
    tools into ordinary LangChain tools. The agent can't tell them from local `@tool`s.
  * resources (read-only data) and prompts (templates) are the other two MCP primitives
  * MCP tools are async, so the agent runs with `await agent.ainvoke(...)`
  * the transport is just config: `stdio` for local servers, `streamable_http` for remote ones

Run:  uv run poe lesson 2.1
"""

import asyncio

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_mcp_adapters.client import MultiServerMCPClient

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.ingest import get_retriever
from docshelp.mcp_server import connections
from docshelp.tools import make_search_docs

QUESTION = "My notes stopped syncing. I'm in the eu-west region. Is something wrong on your side?"


async def explore(llm: BaseChatModel, retriever: BaseRetriever) -> dict:
    client = MultiServerMCPClient(connections())

    header("1. Tools: discovered from the server")
    mcp_tools = await client.get_tools()
    for t in mcp_tools:
        print(f"- {t.name}: {t.description}")

    header("2. Resources: data the app can load")
    page = (await client.get_resources("nimbus_ops", uris="status://page"))[0].as_string()
    print(page)

    header("3. Prompts: reusable templates")
    prompt = await client.get_prompt("nimbus_ops", "incident_reply", arguments={"region": "eu-west"})
    print(prompt[0].text)

    header("4. An agent using MCP tools next to a local tool")
    agent = create_agent(
        llm,
        tools=[*mcp_tools, make_search_docs(retriever)],
        system_prompt="You are DocsHelp, the Nimbus Notes support assistant. " + prompt[0].text,
    )
    print(f"User: {QUESTION}")
    result = await agent.ainvoke({"messages": [{"role": "user", "content": QUESTION}]})
    for message in result["messages"]:
        for call in getattr(message, "tool_calls", []):
            print(f"  -> {call['name']}({call['args']})")
    answer = result["messages"][-1].text
    print(f"DocsHelp: {answer}")
    return {
        "tools": [t.name for t in mcp_tools],
        "resource": page,
        "prompt": prompt,
        "messages": result["messages"],
        "answer": answer,
    }


def run(llm: BaseChatModel, retriever: BaseRetriever) -> dict:
    return asyncio.run(explore(llm, retriever))


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
