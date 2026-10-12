"""Insights: an LLM reads the conversations and tells you what they're about and where the agent fails.

Academy: Monitoring Production Agents M2 "Analyze Trends with Insights"; Building Reliable Agents M3 L2
"Insights Agent"; LangSmith Essentials M4 "Insights".

Concepts:
  * metrics say *that* something changed; reading conversations says *why*, but nobody reads thousands. An
    insights agent does: it summarises each thread (topic, outcome, sentiment), then groups them into
    categories with counts, so you can see "sync failures: 4 threads, 3 unresolved" at a glance.
  * `docshelp.insights` is the local twin, in two LLM steps (`summarize_conversation`, then `cluster`). It
    reads the trace log, so it runs on whatever traffic 9.1-9.7 produced.
  * use it to decide what to fix next, what docs are missing, and which questions should become dataset
    examples (Part 5) or new tools.
  * in LangSmith: project -> Insights -> create a report (pick the traces and the questions you want answered);
    it builds a hierarchy of categories you can drill into, down to individual traces. Paid plans only, hence
    the twin.

Run:  uv run poe lesson 9.8
"""

from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.ingest import get_retriever
from docshelp.insights import insights
from docshelp.monitoring import TRACE_LOG, ensure_traffic, run_filter, summarize


def run(llm: BaseChatModel, retriever: BaseRetriever | None = None, *, path: Path = TRACE_LOG) -> dict:
    production = run_filter('and(has(tags, "production"), eq(status, "success"))')
    summaries = [s for s in summarize(ensure_traffic(llm, retriever, path=path)) if production(s)]
    report = insights(llm, summaries)
    header(f"{len(report['threads'])} conversations, summarised")
    for thread, summary in report["threads"].items():
        print(f"  {thread:22} {summary.outcome:10} {summary.sentiment:8} {summary.topic}")
    header("Categories, most common first")
    for category in report["categories"]:
        print(
            f"  {category['count']:3} threads ({category['unresolved']} unresolved)  {category['name']}: "
            f"{category['description']}"
        )
    return {"threads": len(report["threads"]), "categories": report["categories"]}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
