"""Latency and errors: percentiles instead of averages, where the time goes, and what an incident looks like.

Academy: Monitoring Production Agents M3 "Monitoring Latency and Errors".

Concepts:
  * averages hide the slow tail. p50 (median) is the typical request; p95 and p99 are what your unhappiest
    users wait. Alert on the tail (9.5), not on the mean.
  * where the time goes: a trace's runs split it into model calls and tool calls; the slowest run names say
    what to optimise (a smaller model for routing, caching, fewer tool rounds).
  * errors: the error rate, and errors grouped by type, so one outage reads as one problem, not 40 traces.
  * here an incident is injected on purpose (`simulate_traffic(..., fault=...)`): the model provider times out
    on part of the traffic, as chaos testing would do. The traces record it like any real failure.
  * in LangSmith: latency percentiles and error rate charts on the project's monitoring tab, `gt(latency, 10)`
    or `eq(status, "error")` filters, and the error message on each failed run.

Run:  uv run poe lesson 9.3
"""

from collections import Counter
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.ingest import get_retriever
from docshelp.monitoring import (
    TRACE_LOG,
    build_monitored_agent,
    ensure_traffic,
    load_conversations,
    percentile,
    simulate_traffic,
    summarize,
)
from docshelp.observability import load_rows


def provider_outage(every: int = 3):
    """A fault injector: every `every`-th request times out, as when the model provider has an incident."""
    calls = iter(range(1, 1_000_000))

    def fault(question: str) -> None:
        if next(calls) % every == 0:
            raise TimeoutError("model provider timed out after 30 s")

    return fault


def run(llm: BaseChatModel, retriever: BaseRetriever | None = None, *, path: Path = TRACE_LOG) -> dict:
    ensure_traffic(llm, retriever, path=path)
    header("An incident: the provider times out on every 3rd request")
    agent = build_monitored_agent(llm, retriever)
    incident = set(
        simulate_traffic(
            agent,
            load_conversations(limit=6),
            path=path,
            variant="incident",
            tags=["incident"],
            fault=provider_outage(),
        )
    )
    summaries = summarize(load_rows(path))
    latencies = [s["latency"] for s in summaries if s["status"] == "success"]

    header("Latency percentiles (successful traces)")
    tail = {p: percentile(latencies, p) for p in (50, 95, 99)}
    mean = sum(latencies) / len(latencies) if latencies else 0
    print(f"  mean {mean:.2f} s   p50 {tail[50]:.2f} s   p95 {tail[95]:.2f} s   p99 {tail[99]:.2f} s")

    header("Where the time goes: average per run, by name")
    by_name: dict[str, list[float]] = {}
    for s in summaries:
        for r in s["runs"]:
            if r["run_type"] in ("llm", "tool"):
                by_name.setdefault(f"{r['run_type']}:{r['name']}", []).append((r["latency_ms"] or 0) / 1000)
    breakdown = {name: sum(v) / len(v) for name, v in by_name.items()}
    for name, avg in sorted(breakdown.items(), key=lambda kv: -kv[1]):
        print(f"  {avg:6.2f} s  x{len(by_name[name]):<4} {name}")

    header("Errors")
    errors = [s for s in summaries if s["status"] == "error"]
    rate = len(errors) / len(summaries) if summaries else 0
    print(f"  error rate {rate:.0%} ({len(errors)} of {len(summaries)} traces)")
    kinds = Counter((s["error"] or "").split(":")[0] for s in errors)
    for kind, count in kinds.most_common():
        print(f"  {count:4} x {kind}")
    during = [s for s in summaries if s["trace_id"] in incident]
    incident_rate = sum(s["status"] == "error" for s in during) / len(during) if during else 0
    print(f"  during the incident: {incident_rate:.0%} of requests failed")
    return {
        "percentiles": tail,
        "breakdown": breakdown,
        "error_rate": rate,
        "error_kinds": dict(kinds),
        "incident_error_rate": incident_rate,
    }


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
