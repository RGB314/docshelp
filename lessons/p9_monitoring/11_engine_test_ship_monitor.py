"""Capstone, the improvement loop, part 2: test the proposed fix, ship it only if it's better, then monitor it.

Academy: LangSmith Engine "Test: Testing Engine's Fix", "Deploy: Shipping the Fix", "Monitor"; LangSmith
Essentials M5 "Automating the ADLC with LangSmith Engine", M4 "Capstone"; Monitoring Production Agents
"Capstone Project".

Concepts:
  * test: an experiment (Part 5.6) on the questions that failed in production plus a regression set from the
    offline dataset (so fixing one thing doesn't break another). Old prompt vs candidate, scored by the same
    evaluators that flagged the failures (online evaluators reused offline: `as_offline_evaluator`), plus a
    pairwise judge (Part 5.7) that reads both answers side by side.
  * deploy: `ship_decision` ships only with no metric regressed and a pairwise win. Shipping is moving the
    hub's "prod" tag to the candidate (Part 5.10): the agent pulls "docshelp-support:prod", so there's no
    code deploy, and rollback is moving the tag back.
  * monitor: replay the same conversations with the shipped prompt and compare the online metrics before and
    after. In production this is the dashboard (9.9) and the alerts (9.5) watching the new version.
  * Engine does these steps with a human approving the ship. Here it all runs locally, on any model, for free:
    the whole course in one loop.

Run:  uv run poe lesson 9.10, then: uv run poe lesson 9.11
"""

from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.engine import as_offline_evaluator, find_failures, prompt_target, propose_candidate, ship_decision
from docshelp.evals import averages, make_examples, pairwise_judge, run_experiment
from docshelp.evals import load_rows as load_eval_rows
from docshelp.ingest import get_retriever
from docshelp.monitoring import (
    FEEDBACK_LOG,
    PROMPT_NAME,
    TRACE_LOG,
    OnlineEvaluator,
    build_monitored_agent,
    cites_docs,
    ensure_feedback,
    ensure_prompt,
    ensure_traffic,
    feedback_by_trace,
    helpfulness_evaluator,
    load_conversations,
    prompt_template,
    prompt_text,
    run_filter,
    run_online_evaluators,
    simulate_traffic,
    summarize,
)
from docshelp.observability import load_rows
from docshelp.prompts import prompt_hub

REGRESSION_QUESTIONS = 3  # offline dataset questions added to the failures, so the fix can't break them unnoticed
MONITOR_CONVERSATIONS = 6


def candidate_prompt(llm, retriever, hub, path: Path, feedback_path: Path) -> tuple[str, list[dict]]:
    """The "candidate" version from lesson 9.10, proposing one now if 9.10 hasn't run."""
    production = run_filter('has(tags, "production")')
    summaries = [s for s in summarize(ensure_traffic(llm, retriever, path=path)) if production(s)]
    feedback = ensure_feedback(llm, summaries, feedback_path=feedback_path)
    failures = find_failures(summaries, feedback)
    try:
        return prompt_text(hub, f"{PROMPT_NAME}:candidate"), failures
    except KeyError:
        result = propose_candidate(llm, hub, summaries, feedback)
        return (result["fix"].new_prompt if result["fix"] else prompt_text(hub)), failures


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
    old_prompt = prompt_text(hub)
    new_prompt, failures = candidate_prompt(llm, retriever, hub, path, feedback_path)
    if new_prompt == old_prompt:
        print("The candidate is the prod prompt: nothing to test. Run lesson 9.10 first.")
        return {"shipped": False, "reason": "no candidate"}

    header("3. Test: old vs candidate on the failures plus a regression set")
    questions = list(dict.fromkeys(f["question"] for f in failures))
    questions += [row["question"] for row in load_eval_rows()[:REGRESSION_QUESTIONS]]
    examples = make_examples("docshelp-engine", [({"question": q}, {}) for q in questions])
    evaluators = [as_offline_evaluator(helpfulness_evaluator(llm)), as_offline_evaluator(cites_docs)]
    experiments = {
        name: run_experiment(
            prompt_target(llm, retriever, text),
            examples,
            evaluators,
            prefix=f"engine-{name}",
            dataset_name="docshelp-engine",
        )
        for name, text in (("old", old_prompt), ("new", new_prompt))
    }
    scores = {name: averages(e) for name, e in experiments.items()}
    for name, avg in scores.items():
        print(f"  {name}: " + "  ".join(f"{k} {v:.0%}" for k, v in sorted(avg.items())))
    judge = pairwise_judge(llm)
    wins = {"old": 0, "new": 0, "tie": 0}
    for i, (old_row, new_row) in enumerate(zip(experiments["old"]["rows"], experiments["new"]["rows"])):
        try:
            verdict, _ = judge(
                old_row["question"],
                old_row["outputs"].get("answer", ""),
                new_row["outputs"].get("answer", ""),
                swap=i % 2 == 1,
            )
        except Exception:  # noqa: BLE001 - an unusable verdict counts as a tie, never as a win
            verdict = "tie"
        wins[{"first": "old", "second": "new", "tie": "tie"}[verdict]] += 1
    print(f"  pairwise judge: {wins}")

    header("4. Deploy: move the 'prod' tag, or don't")
    shipped, reason = ship_decision(scores["old"], scores["new"], wins)
    if shipped:
        commit = hub.push(
            PROMPT_NAME,
            prompt_template(new_prompt),
            tags=["prod", "candidate"],
            description="shipped by the improvement loop",
        )
        print(f"  SHIPPED ({reason}): docshelp-support:prod is now {commit}. Roll back by tagging the old commit.")
    else:
        print(f"  NOT shipped: {reason}. Production keeps the old prompt.")

    header("5. Monitor: the same conversations before and after")
    conversations = load_conversations(limit=MONITOR_CONVERSATIONS)
    agent = build_monitored_agent(llm, retriever, system_prompt=prompt_text(hub))
    after_ids = set(simulate_traffic(agent, conversations, path=path, variant="after-fix", tags=["after-fix"]))
    all_traces = summarize(load_rows(path))
    before_threads = {f"{c['id']}/prod" for c in conversations}
    before = [s for s in all_traces if s["metadata"].get("thread_id") in before_threads]
    after = [s for s in all_traces if s["trace_id"] in after_ids]
    checks = [OnlineEvaluator(helpfulness_evaluator(llm)), OnlineEvaluator(cites_docs)]
    feedback = ensure_feedback(llm, before, feedback_path=feedback_path, evaluators=checks)
    feedback += run_online_evaluators(after, checks, feedback_path=feedback_path)
    by_trace = feedback_by_trace(feedback)
    comparison = {}
    for label, group in (("before", before), ("after", after)):
        rates = {
            key: sum(by_trace.get(s["trace_id"], {}).get(key, 0) for s in group) / max(len(group), 1)
            for key in ("helpfulness", "cites_docs")
        }
        comparison[label] = rates
        print(f"  {label:6} {len(group)} traces: " + "  ".join(f"{k} {v:.0%}" for k, v in rates.items()))
    return {
        "questions": len(questions),
        "scores": scores,
        "wins": wins,
        "shipped": shipped,
        "reason": reason,
        "monitor": comparison,
    }


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
