"""Capstone: the DocsHelp support agent (tools + memory + dynamic prompt + structured answer).

Academy: Intro to LangChain M1 L5 project ("Personal Chef"), applied to Nimbus Notes support.

Brings Part 1 together in `src/docshelp/agents.py::build_support_agent`:
  * tools: docs search (RAG) and the billing system's price quote
  * short-term memory: a checkpointer, one thread per support chat
  * runtime context: the customer's name and plan drive a dynamic system prompt
  * `response_format=SupportAnswer`: the final reply is structured (answer, sources, needs_ticket), so
    the app can render citations and escalate to a human when needed

Run:  uv run poe lesson 1.10            (scripted conversation)
      uv run poe lesson 1.10 --chat     (talk to it; empty line to quit)
"""

import argparse
import uuid

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.memory import InMemorySaver

from docshelp.agents import Customer, SupportAnswer, build_support_agent
from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.ingest import get_retriever

CONVERSATION = [
    "How long do deleted notes stay in the trash?",
    "And after that, can an admin still get them back?",  # needs memory to resolve "them"
    "What would 12 seats on Team cost per year, billed annually?",
]


def ask(agent, text: str, config: dict, customer: Customer) -> SupportAnswer:
    result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config, context=customer)
    # Small local models sometimes skip the structured reply; fall back to the plain text answer.
    return result.get("structured_response") or SupportAnswer(answer=result["messages"][-1].text, needs_ticket=False)


def show(reply: SupportAnswer) -> None:
    print(f"DocsHelp: {reply.answer}")
    if reply.sources:
        print(f"  sources: {', '.join(reply.sources)}")
    if reply.needs_ticket:
        print("  -> escalate: a human should follow up (Part 2 adds ticket approval)")


def run(
    llm: BaseChatModel,
    retriever: BaseRetriever,
    conversation: list[str] | None = None,
    customer: Customer | None = None,
) -> list[SupportAnswer]:
    customer = customer or Customer(name="Ada", plan="team")
    agent = build_support_agent(llm, retriever, InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    replies = []
    for turn, text in enumerate(conversation or CONVERSATION, 1):
        header(f"Turn {turn}  ({customer.name}, {customer.plan} plan)")
        print(f"User: {text}")
        replies.append(ask(agent, text, config, customer))
        show(replies[-1])
    return replies


def chat(llm: BaseChatModel, retriever: BaseRetriever) -> None:
    customer = Customer(name="you", plan="team")
    agent = build_support_agent(llm, retriever, InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    while text := input("\nYou: ").strip():
        show(ask(agent, text, config, customer))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat", action="store_true", help="talk to the agent interactively")
    args = parser.parse_args()
    ensure_llm_ready()
    print(f"Using {describe()}")
    if args.chat:
        chat(get_llm(), get_retriever())
    else:
        run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
