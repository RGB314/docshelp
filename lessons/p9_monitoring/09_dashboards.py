"""Dashboards: the monitoring views on one page, built from the trace, feedback and alert logs.

Academy: Introduction to Agent Observability & Evaluations M5 L4 "Monitoring", L5 "Custom Dashboards";
LangSmith Essentials M4 "Dashboards".

Concepts:
  * a dashboard answers "is it healthy?" in ten seconds: volume, error rate, latency percentiles, tokens and
    cost, evaluator scores, security flags, and the same metrics grouped by what you care about (customer plan,
    A/B variant), plus the slowest traces and the alerts that fired.
  * LangSmith: every project has a prebuilt monitoring dashboard; custom dashboards add charts of any metric
    (including feedback keys), filtered with the 9.1 filter language and grouped by metadata or tags. Paid
    features aside, the views are the ones below.
  * the local twin, `docshelp/dashboard.py`, renders the same views as one static HTML file:
    `uv run poe dashboard` writes `data/reports/dashboard.html` from whatever the Part 9 lessons logged.

Run:  uv run poe lesson 9.9      (or just `uv run poe dashboard`)
"""

from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.dashboard import DASHBOARD_PATH, write_dashboard
from docshelp.ingest import get_retriever
from docshelp.monitoring import (
    ALERTS_LOG,
    FEEDBACK_LOG,
    TRACE_LOG,
    AlertRule,
    OnlineEvaluator,
    check_alerts,
    cites_docs,
    ensure_feedback,
    ensure_traffic,
    load_pricing,
    metrics,
    summarize,
)
from docshelp.observability import load_rows


def run(
    llm: BaseChatModel,
    retriever: BaseRetriever | None = None,
    *,
    path: Path = TRACE_LOG,
    feedback_path: Path = FEEDBACK_LOG,
    alerts_path: Path = ALERTS_LOG,
    out: Path = DASHBOARD_PATH,
) -> dict:
    summaries = summarize(ensure_traffic(llm, retriever, path=path))
    feedback = ensure_feedback(llm, summaries, feedback_path=feedback_path, evaluators=[OnlineEvaluator(cites_docs)])
    if not load_rows(alerts_path):  # lesson 9.5 checks alerts; run one rule here so this page works on its own too
        check_alerts(
            summaries, [AlertRule("Error rate above 10%", "error_rate", 0.10)], feedback=feedback, path=alerts_path
        )
    page = write_dashboard(out, trace_log=path, feedback_log=feedback_path, alerts_log=alerts_path)
    overall = metrics(summaries, feedback, load_pricing())
    header("Dashboard written")
    print(f"  {page}")
    print(
        f"  {overall['traces']} traces, error rate {overall['error_rate']:.0%}, p95 {overall['p95_latency']:.2f} s, "
        f"{overall['tokens']:,} tokens, ${overall['cost']:.4f}"
    )
    print("Open it in a browser. Re-run `uv run poe dashboard` any time to refresh it.")
    return {"path": page, "metrics": overall}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
