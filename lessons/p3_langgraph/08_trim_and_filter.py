"""Trim and filter messages inside a graph: three ways to keep the model's context small.

Academy: Intro to LangGraph M2 L4 "Trim and Filter Messages".

Lesson 2.6 did this with middleware on `create_agent`. In your own graph you decide where it happens:
  1. filter in the call: pass only `state["messages"][-3:]` to the model. State keeps everything.
  2. delete from state: a node returns `RemoveMessage(id=...)` for old messages (the `add_messages`
     reducer, lesson 3.6). The checkpointed history really shrinks.
  3. trim by tokens: `trim_messages(..., max_tokens=..., strategy="last", start_on="human")` keeps the
     most recent messages that fit a token budget, never starting mid-exchange. State keeps everything.
Choose 1 or 3 when you may need the full history later (audits, evals); choose 2 to cap storage.

Run:  uv run poe lesson 3.8
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.graph import END, START, MessagesState, StateGraph

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header

PROMPT = "You are DocsHelp, the Nimbus Notes support assistant. Keep replies short."
HISTORY = [
    HumanMessage("Hi, I'm Ada. Our team of twelve uses Nimbus Notes on Windows laptops and iPhones."),
    AIMessage("Hi Ada! Thanks for the context. How can I help your team today?"),
    HumanMessage("Last week we imported about two thousand notes from another app using the importer."),
    AIMessage("Great. The importer keeps notebooks, tags and attachments. Did everything come across?"),
    HumanMessage("Mostly, but a few attachments over 25 MB were skipped, which we fixed by hand."),
    AIMessage("Good fix: attachments are capped at 25 MB on the Team plan. Anything else?"),
    HumanMessage("Yes. Since this morning, notes created on the iPhones don't show up on the laptops."),
    AIMessage("Sorry about that. Does the sync icon on the iPhones show a red slash?"),
    HumanMessage("It does. What should we try first?"),
]


def build(llm: BaseChatModel, strategy: str):
    seen: list[int] = []

    def chat(state: MessagesState) -> dict:
        messages = state["messages"]
        if strategy == "filter":
            messages = messages[-3:]
        elif strategy == "trim":
            recent = trim_messages(
                messages, strategy="last", max_tokens=80, token_counter=count_tokens_approximately, start_on="human"
            )
            messages = [SystemMessage(PROMPT), *recent]
        seen.append(len(messages))
        return {"messages": [llm.invoke(messages)]}

    def cleanup(state: MessagesState) -> dict:
        return {"messages": [RemoveMessage(id=m.id) for m in state["messages"][:-2]]}

    builder = StateGraph(MessagesState)
    builder.add_node(chat)
    if strategy == "delete":
        builder.add_node(cleanup)
        builder.add_edge(START, "cleanup")
        builder.add_edge("cleanup", "chat")
    else:
        builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    return builder.compile(), seen


def run(llm: BaseChatModel) -> dict:
    results = {}
    history = [m.model_copy(update={"id": f"m{i}"}) for i, m in enumerate(HISTORY)]  # ids, so they can be removed
    for strategy in ("filter", "delete", "trim"):
        graph, seen = build(llm, strategy)
        state = graph.invoke({"messages": history})
        results[strategy] = {"model_saw": seen[0], "state_kept": len(state["messages"])}
        header(f"{strategy}: the model saw {seen[0]} messages; the state kept {len(state['messages'])}")
        print(f"DocsHelp: {state['messages'][-1].text}")
    return results


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm())


if __name__ == "__main__":
    main()
