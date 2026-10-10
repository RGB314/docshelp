"""Nimbus Docs: an MCP server that exposes the documentation folder as files (lesson 6.7).

The Academy's MCP research agent reads local files through a filesystem MCP server. This one plays that
role for `data/docs/` with read-only, path-safe tools: list the files, read one, grep across them.
It talks over stdio like the Nimbus Ops server (lesson 2.1).

Run it standalone with `uv run python -m docshelp.docs_mcp_server`.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

DOCS_DIR = Path(__file__).resolve().parents[2] / "data" / "docs"
mcp = FastMCP("nimbus-docs")


@mcp.tool()
def list_docs() -> str:
    """List the documentation files, with each file's first heading."""
    lines = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        lines_of_file = path.read_text(encoding="utf-8").splitlines()
        title = next((ln.lstrip("# ").strip() for ln in lines_of_file if ln.startswith("#")), "")
        lines.append(f"{path.name}: {title}")
    return "\n".join(lines)


@mcp.tool()
def read_doc(name: str) -> str:
    """Read one documentation file by its name, e.g. '03-sync-and-offline.md'."""
    path = (DOCS_DIR / Path(name).name).resolve()  # Path(name).name: no ../ escapes
    if path.parent != DOCS_DIR.resolve() or not path.is_file():
        return f"No such doc: {name}. Use list_docs to see the files."
    return path.read_text(encoding="utf-8")


@mcp.tool()
def grep_docs(pattern: str) -> str:
    """Find lines matching a word or phrase (case-insensitive) across all docs, as 'file: line'."""
    regex = re.compile(re.escape(pattern), re.IGNORECASE)
    hits = [
        f"{path.name}: {line.strip()}"
        for path in sorted(DOCS_DIR.glob("*.md"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if regex.search(line)
    ]
    return "\n".join(hits[:30]) or f"No lines match {pattern!r}."


def connections() -> dict:
    server = {"transport": "stdio", "command": sys.executable, "args": ["-m", "docshelp.docs_mcp_server"]}
    return {"nimbus_docs": server}


if __name__ == "__main__":
    mcp.run()
