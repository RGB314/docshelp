"""Token and cost tracking: what each conversation costs, where the tokens go, and what a hosted model would cost.

Academy: Monitoring Production Agents M1 "Token and Cost Breakdowns", "Tracking Costs", "Viewing Costs".

Concepts:
  * every chat model reports `usage_metadata` (input, output and total tokens) on its reply, and LangChain
    records it on each LLM run of the trace. Cost = input tokens x input price + output tokens x output price;
    output tokens usually cost several times more.
  * prices: `data/pricing.json` maps model names (the `ls_model_name` on each LLM run) to USD per million
    tokens. Local Ollama models are free, so this course's real bill is $0; the "what if" table prices the same
    tokens on illustrative hosted tiers, which is how you'd budget a move to a hosted model.
  * breakdowns: per conversation, per customer plan, input vs output, and per step: an agent turn is several
    model calls (decide, call tools, answer), and each re-sends the growing conversation.
  * in code: `with get_usage_metadata_callback() as cb:` totals tokens per model for any block, without tracing.
  * in LangSmith: token and cost columns on every trace, cost charts on the project dashboard, prices under
    Settings -> Model pricing (matched by model name), and `total_cost` when you `list_runs`.

Run:  uv run poe lesson 9.2
"""

from pathlib import Path

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.customers import Customer
from docshelp.ingest import get_retriever
from docshelp.monitoring import (
    TRACE_LOG,
    build_monitored_agent,
    ensure_traffic,
    load_pricing,
    question_of,
    summarize,
    trace_cost,
)

PER_MILLION = 1_000_000


def run(llm: BaseChatModel, retriever: BaseRetriever | None = None, *, path: Path = TRACE_LOG) -> dict:
    summaries = summarize(ensure_traffic(llm, retriever, path=path))
    pricing = load_pricing()
    llm_runs = [r for s in summaries for r in s["runs"] if r["run_type"] == "llm"]
    input_tokens = sum(r["input_tokens"] or 0 for r in llm_runs)
    output_tokens = sum(r["output_tokens"] or 0 for r in llm_runs)

    header("Tokens and cost per conversation turn")
    for s in summaries[-8:]:
        print(
            f"  {s['total_tokens']:6} tok  {s['llm_calls']} model calls  ${trace_cost(s, pricing):.5f}  {question_of(s)[:55]}"
        )
    actual = sum(trace_cost(s, pricing) for s in summaries)
    print(
        f"\n{len(summaries)} turns, {input_tokens:,} input + {output_tokens:,} output tokens. Your bill: ${actual:.4f}"
    )
    calls = len(llm_runs) / len(summaries) if summaries else 0
    print(f"Each turn makes {calls:.1f} model calls on average: tool use multiplies tokens.")

    header("By customer plan")
    by_plan = {}
    for s in summaries:
        plan = s["metadata"].get("plan", "-")
        by_plan[plan] = by_plan.get(plan, 0) + s["total_tokens"]
    for plan, total in sorted(by_plan.items(), key=lambda kv: -kv[1]):
        print(f"  {plan:12} {total:8,} tokens")

    header("What if this traffic ran on a hosted model? (illustrative prices, data/pricing.json)")
    what_if = {}
    for tier, price in pricing["tiers"].items():
        cost = (input_tokens * price["input"] + output_tokens * price["output"]) / PER_MILLION
        what_if[tier] = cost
        per_thousand = cost / len(summaries) * 1000 if summaries else 0
        print(f"  {tier:24} ${cost:8.4f} for these turns  ->  ${per_thousand:8.2f} per 1,000 turns")

    header("Counting tokens in code: get_usage_metadata_callback()")
    agent = build_monitored_agent(llm, retriever)
    with get_usage_metadata_callback() as usage:
        agent.invoke(
            {"messages": [{"role": "user", "content": "How do I export a notebook to PDF?"}]},
            {"configurable": {"thread_id": "cost-check"}},
            context=Customer(),
        )
    for model, used in usage.usage_metadata.items():  # keyed by the model name the provider reports
        print(f"  {model}: {used.get('input_tokens', 0)} in, {used.get('output_tokens', 0)} out")
    if not usage.usage_metadata:
        print("  (this model reports no model name, so the callback has nothing to key its totals by)")
    return {
        "turns": len(summaries),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": actual,
        "by_plan": by_plan,
        "what_if": what_if,
        "callback": usage.usage_metadata,
    }


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
