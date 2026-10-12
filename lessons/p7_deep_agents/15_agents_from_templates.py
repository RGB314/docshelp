"""Agents from templates: describe an agent in a Markdown file, run it as a deep agent (no new code per agent).

Academy: LangSmith Fleet L1–L9 (local twin); Introduction to Deep Agents M6 "What is MDA?",
"MDA Project Structure".

Concepts:
  * LangSmith Fleet (paid) lets people build agents without code: pick instructions, tools, skills, MCP
    servers, subagents, approvals and a trigger, or start from a template. The twin here is
    `agents/*.md`: YAML frontmatter for the choices, the Markdown body for the instructions.
    `docshelp.agent_templates.build_from_template` turns one into a deep agent. Every field maps to a
    Part 7 concept:
      tools / mcp       what the agent can do (7.3), from a catalog of DocsHelp tools and MCP servers
      approve           human in the loop on those tools (7.4)
      skills            how-to it can read on demand (7.9), given to the run as starting files
      subagents         other templates it can delegate to (7.11)
      model             default or fast (7.2)
      trigger           how it starts: chat, or an inbox on a schedule (an ambient agent, Part 6)
    Computer use (Fleet L8) is the `execute` tool from 7.6 on a sandbox backend.
  * why templates: the agent is data. People review a Markdown diff, not Python; one builder serves every
    agent; a new agent is a new file.
  * Managed Deep Agents (MDA) is LangSmith's hosted version of the same idea for code projects. Its layout:
      agent.py          required: exports the agent (`define_deep_agent`)    here: build_from_template
      instructions.md   the system prompt                                     the template's body
      skills/           one folder per skill                                  skills/
      tools/, middleware/   ordinary Python modules                           src/docshelp/
      schedules/, channels/, connectors/, sandbox/, memory.py, identity.py
                        optional managed capabilities: triggers, chat channels, integrations, a sandbox,
                        memory and auth                                       trigger:, Part 8, 7.6, 7.10
    The instructions and skills are synced to a hosted context store so they can change without a deploy.

Use case: two agents from templates: the refund desk answers a billing complaint, and the inbox watcher
handles the next email in the inbox.

Run:  uv run poe lesson 7.15
"""

import asyncio
import json
from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from docshelp.agent_templates import build_from_template, load_templates, skill_files
from docshelp.ambient import format_email, public
from docshelp.config import PROJECT_ROOT, describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.customers import Customer

COMPLAINT = "We were billed for 12 seats this month but only have 11 members. Can you fix it?"
CUSTOMER = Customer(name="Ada", plan="team", account_id="acct-1001")


def approve(action: dict) -> dict:
    return {"type": "approve"}


async def run_async(
    llm: BaseChatModel,
    retriever: BaseRetriever | None = None,
    *,
    fast_llm: BaseChatModel | None = None,
    review: Callable[[dict], dict] = approve,
) -> dict:
    templates = load_templates()
    header("Templates in agents/")
    for t in templates.values():
        print(f"  {t.name}: tools={t.tools + t.mcp} skills={t.skills} subagents={t.subagents} approve={t.approve}")

    async def run_template(name: str, text: str, thread: str) -> dict:
        template = templates[name]
        agent = await build_from_template(
            template, llm, fast_llm=fast_llm, retriever=retriever, templates=templates, checkpointer=InMemorySaver()
        )
        config = {"configurable": {"thread_id": thread}}
        state = {"messages": [{"role": "user", "content": text}], "files": skill_files(template.skills)}
        result = await agent.ainvoke(state, config, context=CUSTOMER)
        while "__interrupt__" in result:  # the template's `approve` list
            decisions = [review(action) for action in result["__interrupt__"][0].value["action_requests"]]
            print(f"  approval: {[d['type'] for d in decisions]}")
            result = await agent.ainvoke(Command(resume={"decisions": decisions}), config, context=CUSTOMER)
        tools_used = [c["name"] for m in result["messages"] if isinstance(m, AIMessage) for c in m.tool_calls]
        print(f"\n[{name}] tools: {tools_used}\n[{name}] {result['messages'][-1].text}")
        return {"tools": tools_used, "answer": result["messages"][-1].text}

    refund = await run_template("refund-desk", COMPLAINT, "refund-1")

    watcher = templates["inbox-watcher"]
    inbox = PROJECT_ROOT / watcher.trigger["inbox"]
    email = public(json.loads(sorted(inbox.glob("*.json"))[0].read_text(encoding="utf-8")))
    header(f"Trigger: new email in {watcher.trigger['inbox']} (schedule {watcher.trigger['schedule']!r})")
    inbox_result = await run_template("inbox-watcher", format_email(email), email["id"])
    return {"templates": sorted(templates), "refund": refund, "inbox": inbox_result}


def run(llm: BaseChatModel, retriever: BaseRetriever | None = None, **kwargs) -> dict:
    return asyncio.run(run_async(llm, retriever, **kwargs))


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), fast_llm=get_llm(fast=True))


if __name__ == "__main__":
    main()
