"""Deep Agents Code (dcode): a coding agent configured by files in your repo, and what those files mean.

Academy: dcode Essentials ("Welcome to dcode", "dcode Essentials", "Goals", "Skills, MCP, Plugins, and
Marketplaces", "Configuring Subagents", "dcode for Enterprise").

Concepts:
  * dcode is the open-source deep agent for coding in your terminal: the same harness as this part (plan,
    files, shell, subagents, skills, memory), pointed at a repository. Install and run it:
        curl -LsSf https://langch.in/dcode | bash      (see the dcode docs for other ways to install)
        dcode                                          (interactive, in the repo)
        dcode -n "run the tests and summarise failures" -S "uv"   (one task; -S allows shell commands)
    The model is yours to choose (`~/.deepagents/config.toml`); pick a free one to keep it zero cost.
  * goals: `/goal` drafts acceptance criteria from a measurable objective, so a long task has a definition
    of done. `/context` shows how full the context window is.
  * project configuration, checked into this repo under `.deepagents/`:
      AGENTS.md                      project memory, loaded every session        (= `memory=`, lesson 7.10)
      skills/add-a-lesson/SKILL.md   how to add a lesson here                    (= `skills=`, lesson 7.9)
      agents/lesson-reviewer/AGENTS.md  a subagent: frontmatter name/description, body = prompt   (7.11)
      .mcp.json                      MCP servers, in the format Claude Desktop/Code use      (7.3)
    User-level settings live in `~/.deepagents/` (`config.toml` for models, per-agent memory and skills).
  * plugins bundle skills and MCP servers; marketplaces are catalogs to install plugins from (`/plugins`, then
    `/reload`). For enterprise use: shell allowlists (`-S`), approval on risky tools, and tracing to LangSmith.
  * this lesson proves the folder is plain deep-agent configuration: it loads `.deepagents/` with the SDK
    (`docshelp.agent_templates.load_dcode_project`) and builds an equivalent agent over the repo, read-only,
    with `.env` denied entirely, then asks it a question about this repo's conventions.

Run:  uv run poe lesson 7.16
"""

import asyncio
from pathlib import Path

from deepagents import FilesystemPermission
from deepagents.backends import FilesystemBackend
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from docshelp.agent_templates import load_dcode_project, mcp_connections
from docshelp.config import PROJECT_ROOT, describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.deep import budget, deep_agent

QUESTION = (
    "I want to add a lesson to this repo. What must its main() call, and which files do I update besides the lesson?"
)
READ_ONLY = [
    FilesystemPermission(operations=["read", "write"], paths=["/.env", "/.env.*", "/.cache/**"], mode="deny"),
    FilesystemPermission(operations=["write"], paths=["/**"], mode="deny"),
]


async def run_async(llm: BaseChatModel, root: Path = PROJECT_ROOT, *, use_mcp: bool = True) -> dict:
    project = load_dcode_project(root)
    header("What dcode would load from .deepagents/")
    print(f"  memory: AGENTS.md ({len(project.memory.splitlines())} lines)")
    print(f"  skills: {project.skills}")
    print(f"  subagents: {[s['name'] for s in project.subagents]}")
    print(f"  MCP servers: {list(project.mcp_servers)}")

    tools = await MultiServerMCPClient(mcp_connections(project.mcp_servers)).get_tools() if use_mcp else []
    agent = deep_agent(
        llm,
        tools=tools,
        subagents=[{**s, "middleware": [budget()]} for s in project.subagents],
        backend=FilesystemBackend(root_dir=root, virtual_mode=True),
        permissions=READ_ONLY,
        memory=["/.deepagents/AGENTS.md"],
        skills=["/.deepagents/skills/"],
        system_prompt="You are a coding assistant for this repository. Answer from its files.",
    )
    result = await agent.ainvoke({"messages": [{"role": "user", "content": QUESTION}]})
    steps = [c["name"] for m in result["messages"] if isinstance(m, AIMessage) for c in m.tool_calls]
    header(f"Steps: {steps}")
    print(result["messages"][-1].text)
    return {"project": project, "tools": [t.name for t in tools], "steps": steps, "messages": result["messages"]}


def run(llm: BaseChatModel, root: Path = PROJECT_ROOT, *, use_mcp: bool = True) -> dict:
    return asyncio.run(run_async(llm, root, use_mcp=use_mcp))


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm())


if __name__ == "__main__":
    main()
