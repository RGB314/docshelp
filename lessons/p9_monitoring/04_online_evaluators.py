"""Online evaluators: scoring production traces as they arrive, including how customers feel.

Academy: Introduction to Agent Observability & Evaluations M5 L2 "Online Evaluation"; Building Reliable Agents
M3 L3 "Online Evals"; LangSmith Essentials M4 "Online Evaluators"; Monitoring Production Agents M3 "Monitoring
User Sentiment".

Concepts:
  * offline evaluators (Part 5) compare outputs with reference answers. Production has no references, so online
    evaluators judge what they can see: the customer's mood, whether the reply helped, whether it cited a doc.
  * each one is a *rule*: a filter (which traces), a sampling rate (LLM judges cost money: score 10% of a busy
    project, all of a quiet one) and the evaluator. Scores are attached to the trace as feedback, next to user
    thumbs-up/down (Part 5.11), so filters (`and(eq(feedback_key, "helpfulness"), eq(feedback_score, 0))`),
    rules and dashboards can use them.
  * user sentiment is judged from the customer's own words: a frustrated customer is a signal even when the
    answer was right. Watch the negative share over time (9.5 alerts on it).
  * code evaluators (`cites_docs`) are free and instant: run them on everything.
  * in LangSmith: project -> Add rules -> filter + sampling rate + "online evaluator" (an LLM-as-judge prompt
    with a structured score, or Python code); results appear as feedback columns on every scored trace.

Run:  uv run poe lesson 9.4
"""

from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.ingest import get_retriever
from docshelp.monitoring import (
    FEEDBACK_LOG,
    TRACE_LOG,
    OnlineEvaluator,
    cites_docs,
    ensure_traffic,
    feedback_by_trace,
    helpfulness_evaluator,
    question_of,
    run_online_evaluators,
    sentiment_evaluator,
    summarize,
)


def run(
    llm: BaseChatModel,
    retriever: BaseRetriever | None = None,
    *,
    path: Path = TRACE_LOG,
    feedback_path: Path = FEEDBACK_LOG,
    helpfulness_sample: float = 0.5,
) -> dict:
    summaries = [s for s in summarize(ensure_traffic(llm, retriever, path=path)) if s["status"] == "success"]
    evaluators = [
        OnlineEvaluator(sentiment_evaluator(llm), filter='has(tags, "production")'),
        OnlineEvaluator(helpfulness_evaluator(llm), sample_rate=helpfulness_sample),
        OnlineEvaluator(cites_docs),
    ]
    header(f"Scoring {len(summaries)} production traces")
    records = run_online_evaluators(summaries, evaluators, feedback_path=feedback_path)
    scores = feedback_by_trace(records)
    for s in summaries:
        got = scores.get(s["trace_id"], {})
        cells = "  ".join(f"{k}={v:+g}" for k, v in sorted(got.items()))
        print(f"  {cells:52} {question_of(s)[:48]}")

    header("Averages")
    averages = {}
    for key in sorted({r["key"] for r in records}):
        values = [r["score"] for r in records if r["key"] == key]
        averages[key] = sum(values) / len(values)
        print(f"  {key:16} {averages[key]:+.2f}  over {len(values)} traces")
    negative = [s for s in summaries if scores.get(s["trace_id"], {}).get("user_sentiment", 0) < 0]
    print(f"\nUnhappy customers ({len(negative)}):")
    for s in negative:
        print(f"  - {question_of(s)[:90]}")
    counts = {key: sum(r["key"] == key for r in records) for key in averages}
    return {"averages": averages, "counts": counts, "negative": [s["trace_id"] for s in negative]}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
