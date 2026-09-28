"""Entry point for LangGraph Server (`langgraph dev`), referenced from langgraph.json.

No checkpointer is passed: the server provides its own persistence (threads, runs, interrupts).
"""

from docshelp.graph import build_graph

graph = build_graph()
