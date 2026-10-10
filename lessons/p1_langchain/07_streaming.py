"""Streaming: show progress while the model and the agent work.

Academy: LangChain Essentials L3 "Streaming".

Concepts:
  * `llm.stream()` yields message chunks as tokens arrive
  * an agent streams in several modes at once:
      - "updates":  one event per finished step (the model, then the tools, then the model)
      - "messages": LLM tokens from inside the agent, as they are generated
      - "custom":   your own progress events, sent from a tool with `get_stream_writer()`

Real-world use: a chat UI shows "Searching docs..." and then types out the answer.

Run:  uv run poe lesson 1.7
"""

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool
from langgraph.config import get_stream_writer

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.ingest import format_docs, get_retriever


def run(llm: BaseChatModel, retriever: BaseRetriever) -> dict:
    header("1. Token streaming from the model")
    tokens = []
    for chunk in llm.stream("In one sentence: how often do Nimbus notes sync?"):
        tokens.append(chunk.text)
        print(chunk.text, end="|", flush=True)  # "|" marks chunk boundaries
    print()

    @tool
    def search_docs(query: str) -> str:
        """Search the Nimbus Notes product documentation."""
        get_stream_writer()(f"Searching docs for {query!r}...")  # a "custom" stream event
        return format_docs(retriever.invoke(query))

    agent = create_agent(llm, tools=[search_docs], system_prompt="You are DocsHelp. Search the docs first.")

    header("2. Agent streaming: updates + messages + custom, in one pass")
    steps, custom = [], []
    stream = agent.stream(
        {"messages": [{"role": "user", "content": "Can I edit notes offline?"}]},
        stream_mode=["updates", "messages", "custom"],
    )
    for mode, chunk in stream:
        if mode == "updates":
            for step in chunk:
                steps.append(step)
                print(f"\n  [step finished: {step}]")
        elif mode == "custom":
            custom.append(chunk)
            print(f"\n  [progress: {chunk}]")
        else:  # "messages": (token chunk, metadata)
            token, metadata = chunk
            if metadata.get("langgraph_node") == "model" and token.text:
                print(token.text, end="", flush=True)
    print()

    return {"tokens": tokens, "steps": steps, "custom": custom}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
