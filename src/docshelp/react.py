"""A ReAct agent built by hand with LangGraph (Part 3.3): what `create_agent` does under the hood.

        START → assistant ──(tool calls?)──▶ tools
                    ▲  │ no                    │
                    │  ▼                       │
                    │ END                      │
                    └──────────────────────────┘

  * state: `MessagesState`, a `messages` list with the `add_messages` reducer (append, or replace by id)
  * `assistant`: the model with tools bound; it either answers or asks for tool calls
  * `ToolNode`: runs every requested tool call and appends the `ToolMessage`s
  * `tools_condition`: the router; "tools" if the last AI message has tool calls, otherwise END
  * the loop: tools → assistant until the model answers without tool calls

Parts 3.11–3.14 reuse this graph to show streaming, breakpoints, state editing and time travel, and
`server.py` exposes it to LangGraph Studio.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Checkpointer

SUPPORT_PROMPT = (
    "You are DocsHelp, the Nimbus Notes support assistant. Use search_docs for product questions and "
    "quote_price for prices. Never guess."
)


def build_react_agent(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    system_prompt: str = SUPPORT_PROMPT,
    checkpointer: Checkpointer | None = None,
    interrupt_before: list[str] | None = None,
    name: str = "react_agent",
):
    llm_with_tools = llm.bind_tools(list(tools))

    def assistant(state: MessagesState) -> dict:
        # The system prompt is added per call, not stored in state, so it can change between versions.
        return {"messages": [llm_with_tools.invoke([SystemMessage(system_prompt), *state["messages"]])]}

    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(list(tools)))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)  # → "tools" or END
    builder.add_edge("tools", "assistant")
    return builder.compile(checkpointer=checkpointer, interrupt_before=interrupt_before, name=name)
