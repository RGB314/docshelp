"""Moving to production: traffic you didn't choose, and finding the traces that matter with filters.

Academy: Building Reliable Agents M3 L1 "Moving Towards Production"; Introduction to Agent Observability &
Evaluations M5 L1 "Filtering".

Concepts:
  * offline evaluation (Part 5) tests the questions you thought of; production brings the ones you didn't:
    angry customers, prompt injections, card numbers pasted into chat. Monitoring starts with good traces:
    every turn traced, with metadata to slice by (thread, account, plan, A/B variant) and tags ("production").
  * here `data/traffic.json` stands in for real users: `monitoring.simulate_traffic` plays those conversations
    through the support agent (with the launch prompt, `docshelp-support:prod`) and appends each turn to the
    local trace log, `data/monitoring/traces.jsonl`. With a LangSmith key, the same traces also go to your project.
  * a trace log is only useful if you can find things in it. LangSmith's filter language does that, and
    `monitoring.run_filter` runs the same strings locally:
      - `eq(status, "error")`, `gt(latency, 5)`, `gt(total_tokens, 2000)`, `has(tags, "production")`
      - `search("refund")`: full-text search in inputs and outputs
      - metadata: `and(eq(metadata_key, "plan"), eq(metadata_value, "team"))`
    In LangSmith: paste one into the runs filter bar, save it as a view, or use it in code:
    `Client().list_runs(project_name=..., filter=..., is_root=True)`. Rules (9.5) and evaluators (9.4) use
    the same filters to choose which traces they act on.

Run:  uv run poe lesson 9.1
"""

from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from docshelp.config import describe, ensure_llm_ready, get_llm, langsmith_enabled
from docshelp.console import header
from docshelp.ingest import get_retriever
from docshelp.monitoring import (
    TRACE_LOG,
    build_monitored_agent,
    ensure_prompt,
    load_conversations,
    question_of,
    run_filter,
    simulate_traffic,
    summarize,
)
from docshelp.observability import load_rows
from docshelp.prompts import prompt_hub

FILTERS = {
    "errors": 'eq(status, "error")',
    "slower than 5 s": "gt(latency, 5)",
    "over 2,000 tokens": "gt(total_tokens, 2000)",
    "Team plan": 'and(eq(metadata_key, "plan"), eq(metadata_value, "team"))',
    "mentions sync": 'search("sync")',
    "production, over 1 s": 'and(has(tags, "production"), gt(latency, 1))',
}


def run(
    llm: BaseChatModel,
    retriever: BaseRetriever | None = None,
    *,
    path: Path = TRACE_LOG,
    limit: int | None = None,
    hub=None,
) -> dict:
    system_prompt = ensure_prompt(hub or prompt_hub())
    conversations = load_conversations(limit=limit)
    header(f"Production traffic: {len(conversations)} conversations with the prod prompt")
    agent = build_monitored_agent(llm, retriever, system_prompt=system_prompt)
    new = set(simulate_traffic(agent, conversations, path=path))
    summaries = summarize(load_rows(path))
    for s in summaries:
        if s["trace_id"] in new:
            print(f"  {s['status']:7} {s['latency']:6.2f} s {s['total_tokens']:6} tok  {question_of(s)[:70]}")
    print(f"\nThe trace log now holds {len(summaries)} traces ({path.name}).")

    header("Filters: the same strings as LangSmith's filter bar")
    counts = {}
    for label, expression in FILTERS.items():
        matches = [s for s in summaries if run_filter(expression)(s)]
        counts[label] = len(matches)
        print(f"  {len(matches):3}  {label:24} {expression}")
    if langsmith_enabled():  # the same filter against your LangSmith project
        from langsmith import Client

        errors = list(Client().list_runs(filter=FILTERS["errors"], is_root=True, limit=20))
        print(f"\nLangSmith project: {len(errors)} root runs match {FILTERS['errors']}")
    return {"new": len(new), "total": len(summaries), "counts": counts}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
