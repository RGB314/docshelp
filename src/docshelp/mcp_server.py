"""Nimbus Ops: a small MCP server exposing the (pretend) status page to any MCP client.

The Model Context Protocol (MCP) is an open standard for giving AI apps tools and data. A server offers:
  * tools      functions the model can call (here: service status, incidents)
  * resources  read-only data the app can load into context (here: the status page)
  * prompts    reusable prompt templates (here: a customer-facing incident reply)

Any MCP client can use it: DocsHelp through `langchain-mcp-adapters`, Claude Desktop, an IDE, and so on.
It talks over stdio, so the client starts it as a subprocess and nothing touches the network.

Run it standalone with `uv run python -m docshelp.mcp_server` (it then waits for an MCP client on stdin).
"""

from __future__ import annotations

import sys

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("nimbus-ops", log_level="WARNING")  # FastMCP configures logging on creation: keep importers quiet

# Pretend live data. A real server would query the monitoring system.
STATUS = {
    "us-east": "operational",
    "eu-west": "degraded: sync delays of up to 10 minutes (incident INC-2041, fix rolling out)",
    "ap-south": "operational",
}
INCIDENTS = ["INC-2041 (open): delayed sync in eu-west since 09:40 UTC. Notes are safe; they sync once it clears."]


@mcp.tool()
def service_status(region: str) -> str:
    """Current status of the Nimbus Notes service in a data region (us-east, eu-west or ap-south)."""
    status = STATUS.get(region.strip().lower())
    return f"{region}: {status}" if status else f"Unknown region {region!r}. Known regions: {', '.join(STATUS)}."


@mcp.tool()
def list_incidents() -> str:
    """List open incidents affecting Nimbus Notes."""
    return "\n".join(INCIDENTS) or "No open incidents."


@mcp.resource("status://page")
def status_page() -> str:
    """The public status page, as Markdown."""
    return "# Nimbus Notes status\n" + "\n".join(f"- **{region}**: {status}" for region, status in STATUS.items())


@mcp.prompt()
def incident_reply(region: str) -> str:
    """A template for replying to customers affected by an incident in their region."""
    return (
        f"A customer in {region} reports a problem. Check the service status for {region} first. If there is an "
        "incident, apologise, say their notes are safe, and give the incident id. Keep it under 80 words."
    )


def connections() -> dict:
    """How an MCP client starts this server: as a subprocess of the current Python, talking over stdio.

    Remote servers use `{"transport": "streamable_http", "url": "https://..."}` instead; the rest of the
    client code doesn't change.
    """
    return {
        "nimbus_ops": {"transport": "stdio", "command": sys.executable, "args": ["-m", "docshelp.mcp_server"]},
    }


async def load_ops_tools() -> list:
    """The server's tools as LangChain tools (async: each call goes to the server process)."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    return await MultiServerMCPClient(connections()).get_tools()


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
