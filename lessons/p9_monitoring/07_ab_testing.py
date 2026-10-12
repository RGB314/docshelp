"""A/B testing in production: two prompt versions, customers split between them, compared on live metrics.

Academy: Monitoring Production Agents M3 "A/B Testing".

Concepts:
  * offline experiments (Part 5) compare versions on a dataset; an A/B test compares them on real traffic,
    where the questions you didn't anticipate show up. Both are needed: offline first (cheap, safe), then a
    small production split.
  * assignment must be stable: hash the customer id (`assign_variant`), so a customer sees one version, not
    a mix. Record the variant as trace metadata (`variant: "a"`), so every metric can be split by it.
  * the variants here are two versions of the support prompt in the prompt hub (Part 5.10): A is "prod",
    B adds instructions to cite docs and acknowledge frustration. On a server you'd deploy them as two
    assistants (lesson 8.5) and route by customer.
  * compare on the metrics that matter (online evaluator scores, cost, latency) and ask whether the difference
    is real: `two_proportion_test` gives a p-value. With a dozen conversations it rarely is; that's the lesson.
    Decide the metric and the sample size before you look, and run until you reach it.
  * in LangSmith: filter or group the dashboard by `metadata.variant`, or compare the two sets of traces.

Run:  uv run poe lesson 9.7
"""

from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.ingest import get_retriever
from docshelp.monitoring import (
    FEEDBACK_LOG,
    PROMPT_NAME,
    TRACE_LOG,
    OnlineEvaluator,
    assign_variant,
    build_monitored_agent,
    cites_docs,
    ensure_prompt,
    feedback_by_trace,
    helpfulness_evaluator,
    load_conversations,
    metrics,
    prompt_template,
    prompt_text,
    run_online_evaluators,
    simulate_traffic,
    summarize,
    two_proportion_test,
)
from docshelp.observability import load_rows
from docshelp.prompts import prompt_hub

PROMPT_B = (
    "You are DocsHelp, the Nimbus Notes support assistant. Search the docs before answering product questions "
    "and cite the doc file you used, e.g. (03-sync-and-offline.md). If the customer is frustrated, acknowledge "
    "it in one sentence first. Keep answers under 120 words."
)
SPLIT = {"a": 0.5, "b": 0.5}


def run(
    llm: BaseChatModel,
    retriever: BaseRetriever | None = None,
    *,
    path: Path = TRACE_LOG,
    feedback_path: Path = FEEDBACK_LOG,
    hub=None,
) -> dict:
    hub = hub or prompt_hub()
    ensure_prompt(hub)
    hub.push(PROMPT_NAME, prompt_template(PROMPT_B), tags=["variant-b"], description="A/B test: cite docs, empathy")
    prompts = {"a": prompt_text(hub, f"{PROMPT_NAME}:prod"), "b": prompt_text(hub, f"{PROMPT_NAME}:variant-b")}
    agents = {v: build_monitored_agent(llm, retriever, system_prompt=p) for v, p in prompts.items()}

    header("Assigning customers' conversations to variants (stable hash)")
    arms: dict[str, list[dict]] = {"a": [], "b": []}
    for conversation in load_conversations():
        arms[assign_variant(conversation["id"], SPLIT, experiment="prompt-v2")].append(conversation)
    trace_ids = {}
    for variant, conversations in arms.items():
        print(f"  variant {variant}: {len(conversations)} conversations")
        trace_ids[variant] = set(
            simulate_traffic(agents[variant], conversations, path=path, variant=variant, tags=["ab-test"])
        )

    header("Scoring both arms with the same online evaluators")
    summaries = summarize(load_rows(path))
    arm_summaries = {v: [s for s in summaries if s["trace_id"] in ids] for v, ids in trace_ids.items()}
    evaluators = [OnlineEvaluator(helpfulness_evaluator(llm)), OnlineEvaluator(cites_docs)]
    records = run_online_evaluators(
        [s for arm in arm_summaries.values() for s in arm], evaluators, feedback_path=feedback_path
    )
    scores = feedback_by_trace(records)
    results = {}
    for variant, arm in arm_summaries.items():
        m = metrics(arm, records)
        helpful = sum(scores.get(s["trace_id"], {}).get("helpfulness", 0) for s in arm)
        cited = sum(scores.get(s["trace_id"], {}).get("cites_docs", 0) for s in arm)
        results[variant] = {
            "traces": len(arm),
            "helpful": int(helpful),
            "cited": int(cited),
            "tokens": m["tokens"],
            "p50_latency": m["p50_latency"],
        }
        print(
            f"  {variant}: {len(arm)} traces, helpful {int(helpful)}, cited docs {int(cited)}, "
            f"{m['tokens']} tokens, p50 {m['p50_latency']:.2f} s"
        )

    header("Is B really better? (two-proportion z-test on 'helpful')")
    a, b = results["a"], results["b"]
    test = two_proportion_test(a["helpful"], a["traces"], b["helpful"], b["traces"])
    verdict = "significant" if test["p_value"] < 0.05 else "not significant yet: keep the test running"
    print(f"  helpful rate A {test['rate_a']:.0%} vs B {test['rate_b']:.0%}, p = {test['p_value']}: {verdict}")
    return {"arms": {v: len(c) for v, c in arms.items()}, "results": results, "test": test}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
