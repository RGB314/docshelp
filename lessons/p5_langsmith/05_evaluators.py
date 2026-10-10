"""Evaluators: code checks, an LLM judge, and checks on the agent's trajectory.

Academy: Intro to Agent Observability & Evaluations M2 L2 "Evaluators"; Building Reliable Agents
M2 L4 "Code-based Eval", M2 L5 "LLM-as-Judge".

An evaluator takes `inputs`, `outputs` and `reference_outputs` and returns a score (`{"key", "score"}`):
  * code-based: exact, cheap and deterministic. Did retrieval find the right doc? Is a source cited?
    (`src/docshelp/evals.py`)
  * LLM-as-judge: for qualities code can't check, such as "is this answer correct?". `openevals`
    provides tested rubrics (`CORRECTNESS_PROMPT`) and works with any LangChain chat model as the judge.
    Judges can be wrong too: spot-check them against human labels (lesson 5.12).
  * trajectory: an agent can reach a right answer the wrong way (extra calls, the wrong tool).
    `agentevals` compares the tool calls with a reference trajectory (`strict`, `unordered`, `subset`,
    `superset`; arguments compared exactly, ignored, ...) or asks an LLM judge to grade the steps.

Run:  uv run poe lesson 5.5
"""

from agentevals.trajectory.llm import TRAJECTORY_ACCURACY_PROMPT, create_trajectory_llm_as_judge
from agentevals.trajectory.match import create_trajectory_match_evaluator
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.retrievers import BaseRetriever

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.evals import cites_source, expected_source, make_agent_target, make_correctness_judge
from docshelp.ingest import get_retriever

QUESTION = "How long is the workspace export download link valid?"
REFERENCE = {"answer": "7 days.", "source": "05-import-export.md"}
CANDIDATES = {
    "good": {
        "answer": "The download link is valid for 7 days [05-import-export.md].",
        "sources": ["05-import-export.md"],
    },
    "bad": {"answer": "It never expires.", "sources": ["02-plans-and-billing.md"]},
}
PRICE_QUESTION = "What does the Team plan cost for 12 members, billed monthly?"
# The reference trajectory: what a good agent does for the price question (one pricing call).
REFERENCE_TRAJECTORY = [
    HumanMessage(PRICE_QUESTION),
    AIMessage("", tool_calls=[{"name": "quote_price", "args": {}, "id": "ref1", "type": "tool_call"}]),
    ToolMessage("Team plan, 12 members, monthly billing: $96.00 per month.", tool_call_id="ref1"),
    AIMessage("It costs $96.00 per month."),
]


def run(llm: BaseChatModel, retriever: BaseRetriever) -> dict:
    header("1. Code evaluators on a good and a bad answer")
    code = {}
    for label, outputs in CANDIDATES.items():
        code[label] = {**expected_source(outputs, REFERENCE), "cites": cites_source(outputs)["score"]}
        print(f"  {label:5} expected_source={code[label]['score']}  cites_source={code[label]['cites']}")

    header("2. LLM-as-judge: correctness against the reference answer")
    correctness = make_correctness_judge(llm)
    judged = {}
    for label, outputs in CANDIDATES.items():
        result = correctness({"question": QUESTION}, outputs, REFERENCE)
        judged[label] = result["score"]
        print(f"  {label:5} correct={result['score']}  ({result.get('comment') or ''})"[:150])

    header("3. Trajectory: did the agent take the right steps?")
    agent = make_agent_target(llm, retriever)
    outputs = agent({"question": PRICE_QUESTION})
    print(f"  tools called: {outputs['tools']}  answer: {outputs['answer'][:80]}")
    match = create_trajectory_match_evaluator(trajectory_match_mode="superset", tool_args_match_mode="ignore")
    matched = match(outputs=outputs["messages"], reference_outputs=REFERENCE_TRAJECTORY)
    print(f"  match (superset, args ignored): {matched['score']}")
    judge = create_trajectory_llm_as_judge(prompt=TRAJECTORY_ACCURACY_PROMPT, judge=llm)
    graded = judge(outputs=outputs["messages"], reference_outputs=REFERENCE_TRAJECTORY)
    print(f"  LLM trajectory judge: {graded['score']}  ({graded.get('comment') or ''})"[:150])
    return {
        "code": code,
        "correctness": judged,
        "trajectory_match": matched["score"],
        "trajectory_judge": graded["score"],
    }


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
