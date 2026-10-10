"""Dynamic agents: choose the model and the tools per call, at runtime.

Academy: Intro to LangChain M3 L4 "Dynamic Agents".

Concepts:
  * `@wrap_model_call` sees each `ModelRequest` (model, messages, tools, system prompt, state, runtime)
    and can change it with `request.override(...)` before calling `handler`
  * dynamic model: easy questions go to the fast, cheap model; hard ones (SSO, API, errors) to the default
    model. The agent itself is built on the fast model.
  * dynamic tools: Free-plan customers aren't offered `open_ticket` (they get the community forum);
    paid plans are. The model can't call a tool it was never shown.
  * dynamic prompt was lesson 1.9: all three are the same idea, one middleware hook each
  * the same hook can also read custom state: the capstone (2.9) unlocks account tools after verification

Run:  uv run poe lesson 2.8
"""

import re

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, wrap_model_call
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.retrievers import BaseRetriever

from docshelp.agents import plan_aware_prompt
from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.customers import Customer
from docshelp.ingest import get_retriever
from docshelp.tools import make_search_docs, my_account, open_ticket

HARD = re.compile(r"\b(sso|saml|scim|api|webhook|error|crash|data loss|corrupt\w*)\b", re.IGNORECASE)

QUESTIONS = [
    (Customer(name="Sam", plan="free"), "How do I create a new note?"),
    (Customer(name="Ada", plan="team", account_id="acct-1001"), "SAML SSO fails with error 403 for half our users."),
    (Customer(name="Ada", plan="team", account_id="acct-1001"), "How do I rename a notebook?"),
]


def run(llm: BaseChatModel, fast_llm: BaseChatModel, retriever: BaseRetriever) -> dict:
    models: list[str] = []
    offered: dict[str, list[str]] = {}

    @wrap_model_call
    def tools_for_plan(request: ModelRequest, handler):
        customer = request.runtime.context or Customer()
        tools = request.tools if customer.plan != "free" else [t for t in request.tools if t.name != "open_ticket"]
        offered[customer.plan] = [t.name for t in tools]
        return handler(request.override(tools=tools))

    @wrap_model_call
    def model_for_difficulty(request: ModelRequest, handler):
        question = next(m.text for m in reversed(request.messages) if isinstance(m, HumanMessage))
        hard = bool(HARD.search(question))
        models.append("strong" if hard else "fast")
        return handler(request.override(model=llm) if hard else request)

    agent = create_agent(
        fast_llm,  # the default for every call, unless the middleware swaps it
        tools=[make_search_docs(retriever), my_account, open_ticket],
        middleware=[plan_aware_prompt, tools_for_plan, model_for_difficulty],
        context_schema=Customer,
    )
    for customer, question in QUESTIONS:
        header(f"{customer.name} ({customer.plan} plan): {question}")
        calls_before = len(models)
        result = agent.invoke({"messages": [{"role": "user", "content": question}]}, context=customer)
        print(f"  tools offered: {offered[customer.plan]}")
        print(f"  model used:    {', '.join(models[calls_before:])}")
        print(f"DocsHelp: {result['messages'][-1].text}")
    return {"models": models, "tools": offered}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_llm(fast=True), get_retriever())


if __name__ == "__main__":
    main()
