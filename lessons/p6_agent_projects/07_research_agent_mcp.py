"""Deep research, step 3: the same research agent, with its tools served over MCP.

Academy: Project: Deep Research M4 "Research Agent with MCP".

Concepts:
  * the researcher from lesson 6.6 is unchanged. Only its tools differ: they come from the Nimbus Docs
    MCP server (`src/docshelp/docs_mcp_server.py`), which exposes `data/docs/` as files to list, read and
    grep, plus the local `think` tool. Swapping a tool source is configuration, not a rewrite.
  * MCP tools are async, so the graph runs with `ainvoke`. The prebuilt `ToolNode` handles sync and async
    tools alike.
  * reading whole files vs semantic search: file tools give exact, complete text (good for reports that
    must be precise), while the retriever of 6.6 finds meaning across files. Real research agents often
    use both.
  * the server is read-only and path-safe (no `../` escapes): give an agent the least access it needs

Run:  uv run poe lesson 6.7
"""

import asyncio
from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.deep_research import build_researcher, research_input, think
from docshelp.docs_mcp_server import connections

TOPIC = "What does Nimbus Notes offer for importing and exporting notes, and what are the limits?"


async def research(llm: BaseChatModel, mcp_tools: Sequence[BaseTool] | None) -> dict:
    if mcp_tools is None:
        mcp_tools = await MultiServerMCPClient(connections()).get_tools()
    header("Tools from the Nimbus Docs MCP server")
    for t in mcp_tools:
        print(f"  {t.name}: {t.description}")
    researcher = build_researcher(llm, [*mcp_tools, think])
    header(f"Topic: {TOPIC}")
    result = await researcher.ainvoke(research_input(TOPIC))
    return result


def run(llm: BaseChatModel, mcp_tools: Sequence[BaseTool] | None = None) -> dict:
    result = asyncio.run(research(llm, mcp_tools))
    header("Compressed findings")
    print(result["compressed_research"])
    return result


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm())


if __name__ == "__main__":
    main()
