"""Delegation and a subagent team: specialists with their own prompts, tools, models and output formats.

Academy: Introduction to Deep Agents M4 "Delegation", "Building a Subagent Team".

Concepts:
  * `task(description, subagent_type)` runs a subagent on a *fresh* context: it sees only the task
    description, works through as many tool calls as it needs, and returns only its final answer. The main
    agent's context stays small however much digging the subagent did. This is context isolation.
  * Part 2.3's supervisor wrapped each specialist as a tool by hand. Deep agents make it declarative:
      SubAgent          a dict: name, description (how the main agent picks it), system_prompt, tools, and
                        optionally model, middleware, skills, interrupt_on, permissions, response_format
      CompiledSubAgent  any existing graph as a subagent: `{"name", "description", "runnable"}`. Here the
                        hand-built ReAct agent from lesson 3.3 becomes the pricing specialist.
  * `response_format=TicketAnalysis` makes a subagent return structured data, which the main agent receives
    as JSON. Good for anything the caller will process rather than read.
  * the `general-purpose` subagent is always there too (same tools as the main agent), for ad-hoc jobs.

Use case: the main agent asks the ticket analyst for themes, the docs researcher for fixes, and the pricing
agent for a credit amount, then combines them.

Run:  uv run poe lesson 7.11
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.retrievers import BaseRetriever

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.deep import deep_agent, subagent_team
from docshelp.ingest import get_retriever
from docshelp.react import build_react_agent
from docshelp.tools import quote_price

TASK = (
    "Brief me on the ticket queue: get the recurring themes from ticket-analyst, ask docs-researcher what the "
    "docs say about the top theme, and ask pricing what one month's credit for a 12-member Team plan is."
)
PROMPT = "You coordinate Nimbus Notes support specialists. Delegate with the task tool, then write a short brief."


def run(llm: BaseChatModel, retriever: BaseRetriever | None = None, fast_llm: BaseChatModel | None = None) -> dict:
    pricing = {
        "name": "pricing",
        "description": "Quotes exact Nimbus Notes prices for a plan, member count and billing cycle.",
        "runnable": build_react_agent(llm, [quote_price]),  # an existing graph, reused as is
    }
    agent = deep_agent(llm, subagents=[*subagent_team(retriever, fast_llm=fast_llm), pricing], system_prompt=PROMPT)
    result = agent.invoke({"messages": [{"role": "user", "content": TASK}]})

    delegated = [
        (call["args"]["subagent_type"], call["args"]["description"])
        for m in result["messages"]
        if isinstance(m, AIMessage)
        for call in m.tool_calls
        if call["name"] == "task"
    ]
    reports = [m.text for m in result["messages"] if isinstance(m, ToolMessage) and m.name == "task"]
    header("Delegated")
    for (who, what), report in zip(delegated, reports):
        print(f"  -> {who}: {what}\n     <- {report[:160]}")
    header("Brief")
    print(result["messages"][-1].text)
    return {"delegated": [who for who, _ in delegated], "reports": reports, "answer": result["messages"][-1].text}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever(), get_llm(fast=True))


if __name__ == "__main__":
    main()
