"""Agents from configuration: Markdown templates and Deep Agents Code project folders become deep agents.

Two kinds of configuration, one idea: say *what* the agent is (instructions, tools, skills, subagents,
approvals, triggers) in files, and let code turn that into a running agent.

  * templates (lesson 7.15): `agents/*.md`, YAML frontmatter plus instructions. A local twin of LangSmith
    Fleet, where people build agents without code from templates.
  * a Deep Agents Code (`dcode`) project folder (lesson 7.16): `.deepagents/` with `AGENTS.md`, `skills/`,
    `agents/<name>/AGENTS.md` subagents and `.mcp.json`. dcode reads it when it works on this repo; here the
    SDK reads the same files, to show they're plain deep-agent configuration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from deepagents.backends.utils import create_file_data
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool

from docshelp.config import PROJECT_ROOT
from docshelp.customers import Customer
from docshelp.deep import DOCS_READ_ONLY, SKILLS_DIR, budget, deep_agent, list_tickets, search_changelog
from docshelp.tools import make_search_docs, my_account, open_ticket, quote_price, send_reply

TEMPLATES_DIR = PROJECT_ROOT / "agents"


def split_frontmatter(text: str) -> tuple[dict, str]:
    """`---\\n<yaml>\\n---\\n<body>` -> (metadata, body). Files without frontmatter have no metadata."""
    if not text.startswith("---"):
        return {}, text.strip()
    _, meta, body = text.split("---", 2)
    return yaml.safe_load(meta) or {}, body.strip()


# --- Templates (the Fleet twin) -------------------------------------------------------------------------


@dataclass
class AgentTemplate:
    name: str
    description: str
    instructions: str
    model: str = "default"  # or "fast"
    tools: list[str] = field(default_factory=list)
    mcp: list[str] = field(default_factory=list)  # MCP servers by name: nimbus_ops, nimbus_docs
    skills: list[str] = field(default_factory=list)
    subagents: list[str] = field(default_factory=list)  # other templates, by name
    approve: list[str] = field(default_factory=list)  # tools that need a person's OK (HITL)
    trigger: dict = field(default_factory=dict)  # how the agent starts: chat (default), or e.g. an inbox


def load_template(path: Path) -> AgentTemplate:
    meta, body = split_frontmatter(path.read_text(encoding="utf-8"))
    return AgentTemplate(instructions=body, **meta)


def load_templates(directory: Path = TEMPLATES_DIR) -> dict[str, AgentTemplate]:
    templates = [load_template(path) for path in sorted(directory.glob("*.md"))]
    return {t.name: t for t in templates}


def tool_catalog(retriever: BaseRetriever | None = None) -> dict[str, BaseTool]:
    """The tools a template may pick from: Fleet's "what can agents do" list, here DocsHelp's own tools."""
    tools = [
        make_search_docs(retriever),
        search_changelog,
        list_tickets,
        my_account,
        quote_price,
        open_ticket,
        send_reply,
    ]
    return {t.name: t for t in tools}


def skill_files(names: list[str], skills_dir: Path = SKILLS_DIR) -> dict:
    """The chosen skills as starting files for the agent's state (`invoke({"files": ...})`), under /skills/."""
    files = {}
    for name in names:
        for path in sorted((skills_dir / name).iterdir()):
            if path.is_file() and path.suffix in (".md", ".py"):
                files[f"/skills/{name}/{path.name}"] = create_file_data(path.read_text(encoding="utf-8"))
    return files


def skills_backend(skills_dir: Path = SKILLS_DIR) -> CompositeBackend:
    """Files in state, except /skills/, which is the skills folder on disk."""
    skills = FilesystemBackend(root_dir=skills_dir, virtual_mode=True)
    return CompositeBackend(default=StateBackend(), routes={"/skills/": skills})


async def load_mcp_tools(servers: list[str]) -> list[BaseTool]:
    """Tools from the named local MCP servers (lessons 2.1 and 6.7)."""
    if not servers:
        return []
    from langchain_mcp_adapters.client import MultiServerMCPClient

    from docshelp import docs_mcp_server, mcp_server

    known = {**mcp_server.connections(), **docs_mcp_server.connections()}
    return await MultiServerMCPClient({name: known[name] for name in servers}).get_tools()


async def build_from_template(
    template: AgentTemplate,
    llm: BaseChatModel,
    *,
    fast_llm: BaseChatModel | None = None,
    retriever: BaseRetriever | None = None,
    templates: dict[str, AgentTemplate] | None = None,
    checkpointer=None,
    backend=None,
):
    """Turn a template into a deep agent. Pass `skill_files(template.skills)` as the run's `files`.

    On a server, where callers send only messages, pass `backend=skills_backend()` instead: /skills/ is then
    read from the repo's skills/ folder (all of it, read-only), not from the run's files.
    """
    catalog = tool_catalog(retriever)
    storage = {"backend": backend, "permissions": [DOCS_READ_ONLY]} if backend is not None else {}

    def model_for(t: AgentTemplate) -> BaseChatModel:
        return fast_llm if t.model == "fast" and fast_llm is not None else llm

    async def tools_for(t: AgentTemplate) -> list[BaseTool]:
        return [catalog[name] for name in t.tools] + await load_mcp_tools(t.mcp)

    subagents = []
    for name in template.subagents:  # subagent templates become declarative subagents
        sub = (templates or load_templates())[name]
        subagents.append(
            {
                "name": sub.name,
                "description": sub.description,
                "system_prompt": sub.instructions,
                "tools": await tools_for(sub),
                "model": model_for(sub),
                "middleware": [budget()],
            }
        )
    return deep_agent(
        model_for(template),
        tools=await tools_for(template),
        subagents=subagents,
        system_prompt=template.instructions,
        skills=["/skills/"] if template.skills else None,
        interrupt_on={name: True for name in template.approve} or None,
        checkpointer=checkpointer,
        context_schema=Customer,  # tools like my_account read the caller from the run's context
        name=template.name,
        **storage,
    )


# --- A Deep Agents Code project folder (the dcode twin) --------------------------------------------------


@dataclass
class DcodeProject:
    root: Path
    memory: str  # .deepagents/AGENTS.md
    skills: dict[str, str]  # name -> description, from .deepagents/skills/*/SKILL.md
    subagents: list[dict]  # from .deepagents/agents/<name>/AGENTS.md
    mcp_servers: dict  # from .deepagents/.mcp.json


def load_dcode_project(root: Path = PROJECT_ROOT) -> DcodeProject:
    folder = root / ".deepagents"
    skills = {}
    for path in sorted(folder.glob("skills/*/SKILL.md")):
        meta, _ = split_frontmatter(path.read_text(encoding="utf-8"))
        skills[meta["name"]] = meta["description"]
    subagents = []
    for path in sorted(folder.glob("agents/*/AGENTS.md")):
        meta, body = split_frontmatter(path.read_text(encoding="utf-8"))
        subagents.append({"name": meta["name"], "description": meta["description"], "system_prompt": body})
    mcp_file = folder / ".mcp.json"
    servers = json.loads(mcp_file.read_text(encoding="utf-8"))["mcpServers"] if mcp_file.exists() else {}
    return DcodeProject(root, (folder / "AGENTS.md").read_text(encoding="utf-8"), skills, subagents, servers)


def mcp_connections(servers: dict) -> dict:
    """`.mcp.json` entries (the format Claude Desktop and Claude Code also use) as adapter connections."""
    connections = {}
    for name, server in servers.items():
        if "url" in server:
            connections[name] = {"transport": "streamable_http", "url": server["url"]}
        else:
            connections[name] = {"transport": "stdio", "command": server["command"], "args": server.get("args", [])}
    return connections
