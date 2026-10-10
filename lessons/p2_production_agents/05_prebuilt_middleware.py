"""Prebuilt middleware for reliability: retries, call limits and model fallback.

Academy: Intro to LangChain M3 L1 "What is Middleware?" (the built-in middleware it introduces), applied
to the failures a production support agent actually meets.

Concepts:
  * `ToolRetryMiddleware`: retry a failing tool with exponential backoff (a flaky internal API)
  * `ToolCallLimitMiddleware` / `ModelCallLimitMiddleware`: cap tool calls and model calls per run
    (`run_limit`) or per conversation (`thread_limit`), so a looping agent can't burn tokens forever
  * `ModelRetryMiddleware` + `ModelFallbackMiddleware`: retry the model, then fall back to another one
    (an outage, a rate limit, a retired model). Here the "primary" model doesn't exist on purpose.
  * these compose: middleware is a list, applied in order. Other prebuilt ones appear when needed:
    `SummarizationMiddleware` and `ContextEditingMiddleware` (2.6), `HumanInTheLoopMiddleware` (2.7),
    `PIIMiddleware` (Part 9), `TodoListMiddleware` (Part 7).

Run:  uv run poe lesson 2.5
"""

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.ingest import get_retriever
from docshelp.tools import make_search_docs

NO_DELAY = {"initial_delay": 0.0, "jitter": False}  # retry at once; production code keeps the backoff


def ask(agent, question: str) -> dict:
    print(f"User: {question}")
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print(f"DocsHelp: {result['messages'][-1].text}")
    return result


def retries(llm: BaseChatModel) -> int:
    header("1. ToolRetryMiddleware: a flaky tool succeeds on the second attempt")
    attempts = []

    @tool
    def check_sync_service() -> str:
        """Check whether the Nimbus sync service is healthy."""
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("sync health endpoint timed out")
        return "Sync service healthy: 99.98% of syncs succeeded in the last hour."

    agent = create_agent(llm, tools=[check_sync_service], middleware=[ToolRetryMiddleware(max_retries=2, **NO_DELAY)])
    ask(agent, "Is the sync service healthy right now?")
    print(f"  tool attempts: {len(attempts)}")
    return len(attempts)


def limits(llm: BaseChatModel, retriever: BaseRetriever) -> int:
    header("2. Call limits: stop an agent that keeps searching")
    agent = create_agent(
        llm,
        tools=[make_search_docs(retriever)],
        system_prompt="Search the docs as often as you need.",
        middleware=[
            ToolCallLimitMiddleware(tool_name="search_docs", run_limit=2),  # extra calls get an error message
            ModelCallLimitMiddleware(run_limit=3, exit_behavior="end"),  # then the run ends politely
        ],
    )
    result = ask(agent, "Tell me everything about every Nimbus Notes feature, searching each one separately.")
    searches = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    ran = [m for m in searches if m.status != "error"]
    print(f"  searches requested: {len(searches)}, actually run: {len(ran)}")
    return len(ran)


def fallback(primary: BaseChatModel, backup: BaseChatModel) -> str:
    header("3. Model retry, then fallback: the primary model is unavailable")
    agent = create_agent(
        primary,
        tools=[],
        middleware=[
            ModelFallbackMiddleware(backup),  # outermost: if everything inside fails, try the backup
            ModelRetryMiddleware(max_retries=1, on_failure="error", **NO_DELAY),  # retry each model once
        ],
    )
    return ask(agent, "In one sentence, what is Nimbus Notes?")["messages"][-1].text


def run(llm: BaseChatModel, broken_llm: BaseChatModel, retriever: BaseRetriever) -> dict:
    return {
        "attempts": retries(llm),
        "searches_run": limits(llm, retriever),
        "fallback_answer": fallback(broken_llm, llm),
    }


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    # A model name that exists nowhere, standing in for an outage or a retired model.
    run(get_llm(), get_llm("docshelp-no-such-model"), get_retriever())


if __name__ == "__main__":
    main()
