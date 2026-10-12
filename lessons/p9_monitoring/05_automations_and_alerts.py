"""Automations and alerts: let rules route traces to datasets, people and other systems, and page on thresholds.

Academy: Introduction to Agent Observability & Evaluations M5 L3 "Automations"; Building Reliable Agents M3 L4
"Automations"; LangSmith Essentials M4 "Automations", "Annotation Queues"; Monitoring Production Agents M4
"Setting Up Alerts".

Concepts:
  * an automation rule = filter + sampling rate + actions. Actions in LangSmith: add to a dataset, add to an
    annotation queue, run an online evaluator (9.4), send a webhook, extend data retention. Rules run on new
    traces continuously, so the work of 9.1-9.4 happens without anyone looking.
  * here (`monitoring.Rule`, `apply_rules`):
      - failed traces -> the annotation queue (lesson 5.12) and a webhook to the on-call channel
      - answers without a doc citation (feedback `cites_docs` = 0) -> a dataset of examples to fix (9.10 uses it)
      - prompt-injection attempts (tag from 9.6) -> a security webhook
  * alerts are different: they watch a *metric* over a *time window* and fire when it crosses a threshold
    (`AlertRule`): error rate above 10% in 15 minutes, p95 latency above 20 s, any injection attempt, negative
    sentiment above 30%. In LangSmith: project -> Alerts, on error rate, latency or feedback scores, notifying
    PagerDuty or a webhook. Pick thresholds that mean "someone should look now", or people learn to ignore them.

Run:  uv run poe lesson 9.5
"""

from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.ingest import get_retriever
from docshelp.monitoring import (
    ALERTS_LOG,
    FEEDBACK_LOG,
    MONITORING_DIR,
    TRACE_LOG,
    AlertRule,
    OnlineEvaluator,
    Rule,
    add_to_annotation_queue,
    add_to_dataset,
    apply_rules,
    check_alerts,
    cites_docs,
    ensure_feedback,
    ensure_traffic,
    summarize,
    webhook,
    with_feedback,
)

ALERTS = [
    AlertRule("Error rate above 10%", "error_rate", 0.10),
    AlertRule("p95 latency above 20 s", "p95_latency", 20.0),
    AlertRule("Prompt injection attempts", "flagged", 0),
    AlertRule("Negative sentiment above 30%", "negative_rate", 0.30, min_traces=5),
]


def run(
    llm: BaseChatModel,
    retriever: BaseRetriever | None = None,
    *,
    path: Path = TRACE_LOG,
    feedback_path: Path = FEEDBACK_LOG,
    out_dir: Path = MONITORING_DIR,
    alerts_path: Path = ALERTS_LOG,
) -> dict:
    summaries = summarize(ensure_traffic(llm, retriever, path=path))
    # Rules can filter on feedback, so make sure every trace has at least the free code evaluator's score.
    feedback = ensure_feedback(llm, summaries, feedback_path=feedback_path, evaluators=[OnlineEvaluator(cites_docs)])
    traces = with_feedback(summaries, feedback)
    on_call, security = [], []
    rules = [
        Rule(
            "errors to review",
            'eq(status, "error")',
            [add_to_annotation_queue(out_dir / "review_queue.jsonl"), webhook(on_call)],
        ),
        Rule(
            "uncited answers to a dataset",
            'and(eq(feedback_key, "cites_docs"), eq(feedback_score, 0))',
            [add_to_dataset(out_dir / "uncited_answers.jsonl")],
            sample_rate=0.5,
        ),
        Rule("injection attempts to security", 'has(tags, "prompt_injection")', [webhook(security)]),
    ]
    header("Automation rules over the trace log")
    fired = apply_rules(traces, rules)
    by_rule = {rule.name: sum(f["rule"] == rule.name for f in fired) for rule in rules}
    for name, count in by_rule.items():
        print(f"  {count:4} x {name}")
    print(f"  on-call webhook payloads: {len(on_call)}, security webhook payloads: {len(security)}")

    header("Alerts over the last 15 minutes")
    alerts = check_alerts(summaries, ALERTS, feedback=feedback, path=alerts_path)
    for alert in alerts:
        print(f"  FIRING  {alert['alert']}: {alert['metric']}={alert['value']} (threshold {alert['threshold']})")
    if not alerts:
        print("  All quiet.")
    return {
        "rules": by_rule,
        "on_call": len(on_call),
        "security": len(security),
        "alerts": [a["alert"] for a in alerts],
    }


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
