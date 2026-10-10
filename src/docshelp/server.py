"""Entry point for LangGraph Server (`langgraph dev`), referenced from langgraph.json.

Each module-level graph listed in langgraph.json shows up in LangGraph Studio and the server API.
No checkpointer is passed: the server provides its own persistence (threads, runs, interrupts).
"""

from docshelp.config import get_llm
from docshelp.graph import build_graph
from docshelp.react import build_react_agent
from docshelp.tools import quote_price, search_docs

graph = build_graph()  # the DocsHelp support graph (Part 3 capstone)
react_agent = build_react_agent(get_llm(), [search_docs, quote_price])  # the hand-built agent (lesson 3.3)
