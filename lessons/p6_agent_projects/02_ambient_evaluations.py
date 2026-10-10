"""Evaluating the ambient agent: triage accuracy, the right tool calls, and reply quality.

Academy: Project: Ambient Agents M3 "Agent Evaluations".

Concepts (built on Part 5):
  * the inbox files carry labels (`eval`: expected triage, expected tools, success criteria). The agent
    only ever sees `public(email)`, never the labels.
  * evaluate parts separately: triage is a classification, so score it on every email (accuracy, and
    which way it errs: sending a customer to "ignore" is worse than an extra "notify")
  * for emails it should answer: did it call the expected tools (a code check, like a superset
    trajectory match in lesson 5.5), and does the reply meet the email's own success criteria? An LLM
    judge reads the criteria from the reference outputs, so one judge serves every email.
  * both are `run_experiment` calls (lesson 5.6): uploaded to LangSmith if configured, local otherwise

Run:  uv run poe lesson 6.2
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.retrievers import BaseRetriever
from openevals.llm import create_llm_as_judge

from docshelp.ambient import build_inbox_agent, classify_email, load_inbox, public, tools_called
from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.evals import averages, make_examples, run_experiment
from docshelp.ingest import get_retriever

CRITERIA_PROMPT = """You are grading a reply sent by a customer support assistant.

<email>
{inputs}
</email>

<reply>
{outputs}
</reply>

<success_criteria>
{reference_outputs}
</success_criteria>

Does the reply meet every success criterion?"""


def triage_correct(outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "triage_correct", "score": outputs["classification"] == reference_outputs["triage"]}


def expected_tools_called(outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "expected_tools", "score": set(reference_outputs["tools"]) <= set(outputs["tools"])}


def run(llm: BaseChatModel, retriever: BaseRetriever, respond_ids: list[str] | None = None) -> dict:
    inbox = load_inbox()

    header(f"1. Triage on all {len(inbox)} emails")
    examples = make_examples("docshelp-inbox-triage", [(public(e), e["eval"]) for e in inbox])
    triage = run_experiment(
        lambda email: {"classification": classify_email(llm, email).classification},
        examples, [triage_correct], prefix="inbox-triage", dataset_name="docshelp-inbox-triage",
    )
    for row in triage["rows"]:
        got, want = row["outputs"].get("classification"), row["reference"]["triage"]
        print(f"  {'✓' if got == want else '✗'} {row['inputs']['subject'][:50]:50} got={got} want={want}")

    to_answer = [e for e in inbox if e["eval"]["triage"] == "respond"]
    to_answer = [e for e in to_answer if respond_ids is None or e["id"] in respond_ids]
    header(f"2. Responses to the {len(to_answer)} emails that need one")
    agent = build_inbox_agent(llm, retriever)

    def handle(email: dict) -> dict:
        messages = agent.invoke({"email": email})["messages"]
        reply = next((m.text for m in messages if isinstance(m, ToolMessage) and m.name == "send_reply"), "")
        calls = [c for m in messages for c in getattr(m, "tool_calls", [])]
        sent = [c["args"].get("body", "") for c in calls if c["name"] == "send_reply"]
        return {"tools": tools_called(messages), "reply": sent[-1] if sent else reply}

    judge = create_llm_as_judge(prompt=CRITERIA_PROMPT, judge=llm, feedback_key="meets_criteria")

    def meets_criteria(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        return judge(inputs=inputs["body"], outputs=outputs["reply"], reference_outputs=reference_outputs["criteria"])

    examples = make_examples("docshelp-inbox-responses", [(public(e), e["eval"]) for e in to_answer])
    responses = run_experiment(
        handle, examples, [expected_tools_called, meets_criteria], prefix="inbox-responses",
        dataset_name="docshelp-inbox-responses",
    )
    for row in responses["rows"]:
        scores = row["scores"]
        print(f"  {row['inputs']['subject'][:45]:45} tools={scores.get('expected_tools')} "
              f"criteria={scores.get('meets_criteria')}")

    header("Scores")
    summary = {**averages(triage), **averages(responses)}
    for key, value in summary.items():
        print(f"  {key:16} {value:.0%}")
    return {"triage": triage, "responses": responses, "summary": summary}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
