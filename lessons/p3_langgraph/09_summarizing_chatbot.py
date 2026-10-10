"""A chatbot that summarizes its own history, built by hand in LangGraph.

Academy: Intro to LangGraph M2 L5 "Chatbot w/ Summarizing Messages and Memory".

Concepts (the graph is `src/docshelp/chatbot.py::build_summarizing_chatbot`):
  * extra state next to `messages`: a running `summary` string
  * a conditional edge decides *when* to summarize (more than 6 messages)
  * the summarize node extends the previous summary, then deletes old messages with `RemoveMessage`
  * the conversation node puts the summary in the system prompt, so the model still knows old facts
  * a checkpointer keeps both `summary` and `messages` per thread. This is what
    `SummarizationMiddleware` (lesson 2.6) does for `create_agent`, written out in LangGraph.

Run:  uv run poe lesson 3.9
"""

import uuid

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver

from docshelp.chatbot import build_summarizing_chatbot
from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header

CHAT = [
    "Hi, I'm Ada from Acme Research. We're on the Team plan.",
    "We all use Windows laptops.",
    "Our sync icon has a red slash since this morning.",
    "I already tried signing out and back in.",
    "What else can I try?",
]


def run(llm: BaseChatModel) -> dict:
    bot = build_summarizing_chatbot(llm, InMemorySaver(), max_messages=6, keep=2)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    sizes = []
    for text in CHAT:
        header(f"User: {text}")
        for update in bot.stream({"messages": [{"role": "user", "content": text}]}, config, stream_mode="updates"):
            print(f"  ran: {', '.join(update)}")
        state = bot.get_state(config).values
        sizes.append(len(state["messages"]))
        print(f"DocsHelp: {state['messages'][-1].text}")
        print(f"  [{sizes[-1]} messages kept; summary: {state.get('summary') or '(none yet)'}]")
    return {"sizes": sizes, "summary": bot.get_state(config).values.get("summary", "")}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm())


if __name__ == "__main__":
    main()
