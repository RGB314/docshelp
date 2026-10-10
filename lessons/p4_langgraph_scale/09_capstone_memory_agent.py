"""Capstone: a support agent that remembers each customer across conversations.

Academy: Intro to LangGraph M5 L5 "Agent with Long-Term Memory" (task_mAIstro), applied to support.

Brings Part 4 together in `src/docshelp/memory.py::build_memory_agent`:
  * three kinds of memory per customer, in the Store: a profile (one document, lesson 4.7), past issues
    (a collection, 4.8) and instructions (free text: how they like to be helped)
  * the model decides *when* to remember: it calls an `UpdateMemory` tool with the kind of update, and
    an `act` node runs it with a trustcall extractor (or a plain prompt for instructions)
  * every turn starts by recalling all three into the system prompt, so a brand-new thread already
    knows the customer
  * short-term memory (checkpointer, per thread) and long-term memory (store, per customer) side by side

Run:  uv run poe lesson 4.9
"""

import uuid

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever
from langgraph.store.memory import InMemoryStore

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.customers import Customer
from docshelp.ingest import get_retriever
from docshelp.memory import build_memory_agent, recall

ADA = Customer(name="Ada", plan="team", account_id="acct-1001")
THREAD_1 = "Hi, I'm Ada, the workspace admin. Notes created on my iPhone take ages to show up on my Windows laptop."
THREAD_2 = "Hello again! Is there any news about my problem?"


def talk(agent, text: str) -> list:
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    print(f"User: {text}")
    messages = agent.invoke({"messages": [{"role": "user", "content": text}]}, config, context=ADA)["messages"]
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                print(f"  -> {call['name']}({call['args']})")
    print(f"DocsHelp: {messages[-1].text}")
    return messages


def run(llm: BaseChatModel, retriever: BaseRetriever, store: InMemoryStore | None = None) -> dict:
    store = store or InMemoryStore()
    agent = build_memory_agent(llm, store, retriever)

    header("Thread 1: a new customer describes a problem")
    talk(agent, THREAD_1)
    header("What DocsHelp now remembers about Ada")
    print(recall(store, ADA.account_id))

    header("Thread 2: a new conversation, days later")
    answer = talk(agent, THREAD_2)[-1].text
    return {"memory": recall(store, ADA.account_id), "answer": answer, "store": store}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
