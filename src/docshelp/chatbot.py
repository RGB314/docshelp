"""A support chatbot that summarizes its own history, built by hand (Parts 3.9 and 3.10).

        START → conversation ──(more than `max_messages`?)──▶ summarize → END
                                     │ no
                                     ▼
                                    END

Lesson 2.6 used `SummarizationMiddleware`; this is the same idea one layer down, in plain LangGraph:
  * a `summary` key in state, next to `messages`
  * `conversation` adds the summary to the system prompt, so older turns stay "known"
  * `summarize` extends the summary, then deletes all but the last `keep` messages with `RemoveMessage`
    (the `add_messages` reducer removes messages by id)
With a durable checkpointer (lesson 3.10), both the summary and the recent messages survive restarts.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Checkpointer

PROMPT = "You are DocsHelp, the Nimbus Notes support assistant. Keep replies short and friendly."


class ChatState(MessagesState):
    summary: str


def build_summarizing_chatbot(
    llm: BaseChatModel,
    checkpointer: Checkpointer | None = None,
    *,
    max_messages: int = 6,
    keep: int = 2,
):
    def conversation(state: ChatState) -> dict:
        summary = state.get("summary", "")
        system = PROMPT + (f"\n\nSummary of the conversation so far: {summary}" if summary else "")
        return {"messages": [llm.invoke([SystemMessage(system), *state["messages"]])]}

    def should_summarize(state: ChatState) -> Literal["summarize", "__end__"]:
        return "summarize" if len(state["messages"]) > max_messages else END

    def summarize(state: ChatState) -> dict:
        summary = state.get("summary", "")
        ask = (
            f"This is the summary so far: {summary}\n\nExtend it with the new messages above."
            if summary
            else "Summarize the conversation above."
        ) + " Keep every customer fact: name, plan, devices, problems and what was tried."
        new_summary = llm.invoke([*state["messages"], HumanMessage(ask)]).text
        old = state["messages"][:-keep]
        return {"summary": new_summary, "messages": [RemoveMessage(id=m.id) for m in old]}

    builder = StateGraph(ChatState)
    builder.add_node(conversation)
    builder.add_node(summarize)
    builder.add_edge(START, "conversation")
    builder.add_conditional_edges("conversation", should_summarize)
    builder.add_edge("summarize", END)
    return builder.compile(checkpointer=checkpointer, name="summarizing_chatbot")
