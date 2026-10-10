"""Managing long conversations: summarize, trim, and clear old tool results.

Academy: Intro to LangChain M3 L2 "Managing Long Conversations".

Every turn resends the whole history (lesson 1.8), so long support chats get slow, expensive and
eventually exceed the context window. Three remedies, each a middleware:

  * `SummarizationMiddleware`: when the history passes a `trigger` (messages, tokens or a fraction of
    the context window), a cheaper model summarizes the old part and the summary replaces it in state.
    The last `keep` messages stay verbatim. Old details are compressed, not lost.
  * trimming with a custom `@before_model` hook: drop all but the most recent messages from state with
    `RemoveMessage(id=REMOVE_ALL_MESSAGES)`. Cheap and predictable, but old details are gone for good.
  * `ContextEditingMiddleware` + `ClearToolUsesEdit`: replace old tool *results* (often the bulkiest part)
    with "[cleared]" in what the model sees, leaving the saved state untouched

Run:  uv run poe lesson 2.6
"""

import uuid

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    SummarizationMiddleware,
    before_model,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, RemoveMessage
from langchain_core.messages.utils import trim_messages
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.ingest import get_retriever
from docshelp.tools import make_search_docs

PROMPT = "You are DocsHelp, the Nimbus Notes support assistant. Keep replies short."
CHAT = [
    "Hi, I'm Ada from Acme Research, we're on the Team plan.",
    "I use Windows 11.",
    "We have 12 seats.",
    "My notes stopped syncing this morning. What should I try?",
]


def chat(agent, turns: list[str]) -> tuple[list[int], list[BaseMessage]]:
    """Run a conversation on one thread. Returns the thread's size after each turn, and its final messages."""
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    sizes, messages = [], []
    for text in turns:
        messages = agent.invoke({"messages": [{"role": "user", "content": text}]}, config)["messages"]
        sizes.append(len(messages))
        print(f"User: {text}\nDocsHelp: {messages[-1].text}   [{sizes[-1]} messages saved]")
    return sizes, messages


def summarization(llm: BaseChatModel, fast_llm: BaseChatModel) -> tuple[list[int], str]:
    header("1. SummarizationMiddleware: compress old turns into a summary")
    agent = create_agent(
        llm,
        system_prompt=PROMPT,
        # Summarize once the thread reaches 6 messages, keeping the last 2 as they are.
        middleware=[SummarizationMiddleware(model=fast_llm, trigger=("messages", 6), keep=("messages", 2))],
        checkpointer=InMemorySaver(),
    )
    sizes, messages = chat(agent, CHAT)
    summary = messages[0].text  # the summary now stands in for the older turns
    print(f"\nThread now starts with:\n{summary}")
    return sizes, summary


def trimming(llm: BaseChatModel) -> list[int]:
    header("2. Trimming in a before_model hook: keep only the latest messages")
    seen: list[int] = []

    @before_model
    def keep_recent(state: AgentState, runtime) -> dict:
        # `token_counter=len` counts messages; start the window on a user message so it reads naturally.
        recent = trim_messages(state["messages"], strategy="last", max_tokens=4, token_counter=len, start_on="human")
        seen.append(len(recent))
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *recent]}

    agent = create_agent(llm, system_prompt=PROMPT, middleware=[keep_recent], checkpointer=InMemorySaver())
    chat(agent, CHAT[:3])
    print(f"\nMessages the model saw per turn: {seen}")
    return seen


def context_editing(llm: BaseChatModel, retriever: BaseRetriever) -> None:
    header("3. ContextEditingMiddleware: clear old tool results from the model's view")
    agent = create_agent(
        llm,
        tools=[make_search_docs(retriever)],
        system_prompt=PROMPT + " Search the docs for each topic separately.",
        # Once tool results pass ~300 tokens, keep only the latest one; older ones become "[cleared]".
        middleware=[ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=300, keep=1)])],
    )
    question = "Compare offline mode, sharing and export, searching for each separately."
    print(f"User: {question}")
    print(f"DocsHelp: {agent.invoke({'messages': [{'role': 'user', 'content': question}]})['messages'][-1].text}")


def run(llm: BaseChatModel, fast_llm: BaseChatModel, retriever: BaseRetriever) -> dict:
    summarized_seen, summary = summarization(llm, fast_llm)
    trimmed_seen = trimming(llm)
    context_editing(llm, retriever)
    return {"summarized_seen": summarized_seen, "summary": summary, "trimmed_seen": trimmed_seen}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_llm(fast=True), get_retriever())


if __name__ == "__main__":
    main()
