"""Tools and `create_agent`: the model decides which functions to call, and in what order.

Academy: LangChain Essentials L1 "Create Agent" and L4 "Tools", Intro to LangChain M1 L2 "Tools".

Concepts:
  * `@tool` turns a typed, documented function into a tool; its name, docstring and type hints become
    the schema the model sees
  * tool calling: the model only *asks* for a call (`AIMessage.tool_calls`); your code runs it
  * `create_agent`: the loop "model -> run requested tools -> model ..." until the model answers.
    It returns a compiled LangGraph graph (Part 3 builds such graphs by hand).

Run:  uv run poe lesson 1.6
"""

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.ingest import get_retriever
from docshelp.tools import make_search_docs, quote_price

QUESTION = (
    "We're a team of 12 and need SSO with Okta. Which plan do we need, and what would the "
    "Team plan cost us per year with annual billing?"
)


def run(llm: BaseChatModel, retriever: BaseRetriever) -> dict:
    search_docs = make_search_docs(retriever)

    header("1. A tool is a function plus a schema")
    print(f"name: {quote_price.name}\ndescription: {quote_price.description}\nargs: {quote_price.args}")
    print(f"\nCalled directly: {quote_price.invoke({'plan': 'team', 'members': 5})}")

    header("2. Tool calling by hand: the model asks, your code runs the tool")
    request = llm.bind_tools([quote_price]).invoke("Price for 12 people on Team, billed annually?")
    manual_call = request.tool_calls[0] if request.tool_calls else {}
    print(f"Model requested: {manual_call}")
    if manual_call:
        print(f"Tool result:     {quote_price.invoke(manual_call['args'])}")

    header("3. create_agent runs that loop for you")
    agent = create_agent(
        llm,
        tools=[search_docs, quote_price],
        system_prompt="You are DocsHelp, the Nimbus Notes support assistant. Use tools; don't guess.",
    )
    print(f"Q: {QUESTION}\n")
    result = agent.invoke({"messages": [{"role": "user", "content": QUESTION}]})
    tools_called = [
        call["name"] for msg in result["messages"] if isinstance(msg, AIMessage) for call in msg.tool_calls
    ]
    for msg in result["messages"]:
        for call in getattr(msg, "tool_calls", []) or []:
            print(f"  -> tool call: {call['name']}({call['args']})")
    answer = result["messages"][-1].text
    print(f"\nA: {answer}")

    return {"manual_call": manual_call, "tools_called": tools_called, "messages": result["messages"], "answer": answer}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
