"""Short- vs long-term memory: the LangGraph Store remembers a customer across conversations.

Academy: Intro to LangGraph M5 L1 "Short vs Long-Term Memory", M5 L2 "LangGraph Store".

Concepts:
  * short-term memory = the checkpointer: one thread's messages and state. A new thread starts empty.
  * long-term memory = a Store: JSON documents under a *namespace* (a tuple, e.g. ("facts", account))
    and a key, shared by every thread. `put`, `get`, `search`, `delete`, `list_namespaces`.
  * pass both at compile time: `compile(checkpointer=..., store=...)`. Nodes reach the store through
    `runtime.store`, and the runtime context (lesson 2.2) tells them whose memory to use.
  * when to write: here a `write_memory` node runs after every reply ("in the hot path"). Lesson 4.9 lets
    the model decide when; writing in the background, after the conversation, is a third option.
  * the LangChain layer: `create_agent(..., store=store)` gives tools `runtime.store` too

Run:  uv run poe lesson 4.5
"""

import uuid

from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.customers import Customer
from docshelp.memory import ns

ADA = Customer(name="Ada", plan="team", account_id="acct-1001")


def build_chatbot(llm: BaseChatModel, store: InMemoryStore):
    def call_model(state: MessagesState, runtime: Runtime[Customer]) -> dict:
        facts = runtime.store.get(ns("facts", runtime.context.account_id), "facts")
        system = "You are DocsHelp, the Nimbus Notes support assistant. Keep replies short."
        system += f"\nWhat you remember about this customer: {facts.value['text'] if facts else 'nothing yet'}"
        return {"messages": [llm.invoke([SystemMessage(system), *state["messages"]])]}

    def write_memory(state: MessagesState, runtime: Runtime[Customer]) -> dict:
        namespace = ns("facts", runtime.context.account_id)
        old = runtime.store.get(namespace, "facts")
        prompt = (
            f"Known facts about the customer: {old.value['text'] if old else 'none'}\n"
            "Update them with anything new from the conversation below. Reply with the facts only, one line.\n\n"
            + "\n".join(f"{m.type}: {m.text}" for m in state["messages"])
        )
        runtime.store.put(namespace, "facts", {"text": llm.invoke(prompt).text.strip()})
        return {}

    builder = StateGraph(MessagesState, context_schema=Customer)
    builder.add_node(call_model)
    builder.add_node(write_memory)
    builder.add_edge(START, "call_model")
    builder.add_edge("call_model", "write_memory")
    builder.add_edge("write_memory", END)
    return builder.compile(checkpointer=InMemorySaver(), store=store)


@tool
def remember_preference(preference: str, runtime: ToolRuntime[Customer]) -> str:
    """Remember how the customer likes to be helped, for future conversations."""
    runtime.store.put(ns("preferences", runtime.context.account_id), str(uuid.uuid4()), {"text": preference})
    return f"Saved: {preference}"


def chat(bot, text: str) -> str:
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}  # a brand-new conversation every time
    reply = bot.invoke({"messages": [{"role": "user", "content": text}]}, thread, context=ADA)["messages"][-1].text
    print(f"User: {text}\nDocsHelp: {reply}")
    return reply


def run(llm: BaseChatModel) -> dict:
    store = InMemoryStore()
    bot = build_chatbot(llm, store)

    header("1. Thread 1: the customer introduces themselves")
    chat(bot, "Hi, I'm Ada, a workspace admin. We all use Windows laptops.")
    print(f"  store now holds: {store.get(ns('facts', ADA.account_id), 'facts').value}")

    header("2. Thread 2: a new conversation, empty short-term memory")
    chat(bot, "Which operating system do we use again?")

    header("3. The same Store from create_agent: a tool writes through runtime.store")
    agent = create_agent(llm, tools=[remember_preference], context_schema=Customer, store=store)
    text = "Please always answer me in bullet points."
    print(f"User: {text}")
    reply = agent.invoke({"messages": [{"role": "user", "content": text}]}, context=ADA)["messages"][-1].text
    print(f"DocsHelp: {reply}")

    header("What the store holds, by namespace")
    for namespace in store.list_namespaces():
        for item in store.search(namespace):
            print(f"  {namespace} / {item.key[:8]}: {item.value}")
    return {
        "facts": store.get(ns("facts", ADA.account_id), "facts").value["text"],
        "preferences": [i.value["text"] for i in store.search(ns("preferences", ADA.account_id))],
    }


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm())


if __name__ == "__main__":
    main()
