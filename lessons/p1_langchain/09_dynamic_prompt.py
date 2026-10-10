"""Dynamic prompts: build the system prompt per call from runtime context (the customer's plan).

Academy: LangChain Essentials L8 "Dynamic Prompt".

Concepts:
  * runtime context: facts your app knows about the caller (user, plan, locale), passed with
    `agent.invoke(..., context=...)` and typed by `context_schema`. The model never has to be told them
    by the user, and they can't be spoofed by the user.
  * `@dynamic_prompt` is middleware (Part 2 covers middleware in depth): it rewrites the system prompt
    before every model call
  * one agent, many behaviours: Free users hear about upgrades; Enterprise users about their account manager

Run:  uv run poe lesson 1.9
"""

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_core.language_models import BaseChatModel

from docshelp.agents import Customer, support_prompt
from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header

QUESTION = "Can we turn on single sign-on?"


@dynamic_prompt
def plan_aware_prompt(request: ModelRequest) -> str:
    return support_prompt(request.runtime.context)


def run(llm: BaseChatModel) -> list[str]:
    agent = create_agent(llm, tools=[], middleware=[plan_aware_prompt], context_schema=Customer)
    replies = []
    for customer in (Customer(name="Sam", plan="free"), Customer(name="Grace", plan="enterprise")):
        header(f"{customer.name} ({customer.plan} plan) asks: {QUESTION}")
        print(f"System prompt: {support_prompt(customer)}\n")
        result = agent.invoke({"messages": [{"role": "user", "content": QUESTION}]}, context=customer)
        replies.append(result["messages"][-1].text)
        print(f"DocsHelp: {replies[-1]}")
    return replies


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm())


if __name__ == "__main__":
    main()
