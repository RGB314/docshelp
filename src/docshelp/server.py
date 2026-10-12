"""Entry point for LangGraph Server (`langgraph dev`), referenced from langgraph.json.

Each module-level graph listed in langgraph.json shows up in LangGraph Studio and the server API.
No checkpointer or store is passed: the server provides its own persistence (threads, runs, interrupts,
long-term memory), which replaces whatever a graph was compiled with.
"""

import asyncio

from langchain_core.runnables import RunnableConfig

from docshelp.agent_templates import build_from_template, load_templates, skills_backend
from docshelp.ambient import build_inbox_agent
from docshelp.config import get_llm
from docshelp.deep import build_docs_researcher, build_ops_analyst
from docshelp.deployment import build_deployed_agent, build_inbox_ingest
from docshelp.graph import build_graph
from docshelp.memory import build_memory_agent
from docshelp.react import build_react_agent
from docshelp.tools import quote_price, search_docs

graph = build_graph()  # the DocsHelp support graph (Part 3 capstone)
react_agent = build_react_agent(get_llm(), [search_docs, quote_price])  # the hand-built agent (lesson 3.3)
support_agent = build_deployed_agent(get_llm(), get_llm(fast=True))  # one graph, many assistants (lesson 8.5)
memory_agent = build_memory_agent(get_llm(), store=None, on_server=True)  # Part 4's agent on the server's Store (8.7)
inbox_agent = build_inbox_agent(get_llm(), hitl=True, memory=True, on_server=True)  # Part 6's ambient agent (8.12)
inbox_ingest = build_inbox_ingest()  # what a cron runs to feed it new emails (lesson 8.12)
docs_researcher = build_docs_researcher(get_llm())  # the async subagent's worker (lesson 7.13)
ops_analyst = build_ops_analyst(get_llm(), on_server=True)  # the Deep Agents capstone (lessons 7.14, 8.8)


async def agent_from_template(config: RunnableConfig):
    """A graph *factory*: the server calls it for each run and uses the graph it returns (lesson 8.9).

    The assistant's config names a Markdown template from agents/, so deploying a new agent takes no code:
    write a template, then create an assistant with `{"configurable": {"template": "<name>"}}`.
    """
    templates = await asyncio.to_thread(load_templates)  # file reads: keep them off the server's event loop
    name = config.get("configurable", {}).get("template", "refund-desk")
    if name not in templates:
        raise ValueError(f"No template {name!r} in agents/. Choose one of: {', '.join(templates)}")
    llm, fast_llm, backend = await asyncio.to_thread(lambda: (get_llm(), get_llm(fast=True), skills_backend()))
    return await build_from_template(templates[name], llm, fast_llm=fast_llm, templates=templates, backend=backend)
