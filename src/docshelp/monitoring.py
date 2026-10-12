"""Watching DocsHelp in production: traffic, traces, metrics, online evaluators, automations and alerts (Part 9).

Part 5 taught tracing and offline evaluation: you choose the inputs. In production the inputs choose you, so
the questions change: what does it cost, how slow is it, where does it fail, are customers happy, and is the
new prompt better than the old one? Everything here runs on the local trace log (`observability.traced`)
and mirrors a LangSmith feature, which the lessons name:

  * `simulate_traffic`: production-like conversations (data/traffic.json) through the support agent, each turn
    traced with metadata (thread, account, plan, variant). Stands in for real users.
  * `summarize`: one row per trace, with the fields LangSmith shows in its runs table
  * `run_filter`: LangSmith's filter language (`and(eq(status, "error"), gt(latency, 5))`) on those rows
  * costs and latency percentiles; online evaluators that write feedback; automation rules; alert rules
  * A/B variant assignment and a significance test
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from docshelp.config import DATA_DIR, structured_llm
from docshelp.customers import Customer
from docshelp.observability import append_rows, load_rows, record_feedback, traced
from docshelp.tools import make_search_docs, my_account, quote_price

TRAFFIC_PATH = DATA_DIR / "traffic.json"
PRICING_PATH = DATA_DIR / "pricing.json"
MONITORING_DIR = DATA_DIR / "monitoring"  # git-ignored: everything the lessons generate
TRACE_LOG = MONITORING_DIR / "traces.jsonl"
FEEDBACK_LOG = MONITORING_DIR / "feedback.jsonl"
ALERTS_LOG = MONITORING_DIR / "alerts.jsonl"

# The prompt in production. Deliberately thin: it says nothing about citing docs or about upset customers,
# so the monitoring lessons have real problems to find, and the improvement loop (9.10-9.11) something to fix.
PROMPT_NAME = "docshelp-support"
SUPPORT_PROMPT_V1 = "You are DocsHelp, the Nimbus Notes support assistant. Answer the customer's question."


# --- The agent in production, and its traffic -------------------------------------------------------------


def build_monitored_agent(
    llm: BaseChatModel, retriever: BaseRetriever | None = None, *, system_prompt: str = SUPPORT_PROMPT_V1, middleware=()
):
    """The support agent as deployed: tools, a system prompt (from the prompt hub), optional guard middleware."""
    return create_agent(
        llm,
        tools=[make_search_docs(retriever), quote_price, my_account],
        system_prompt=system_prompt,
        middleware=list(middleware),
        context_schema=Customer,
        checkpointer=InMemorySaver(),
        name="docshelp_support",
    )


def prompt_template(text: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("system", text)])


def prompt_text(hub, ref: str = f"{PROMPT_NAME}:prod") -> str:
    """The system prompt text of a version in the prompt hub (Part 5.10), e.g. "docshelp-support:prod"."""
    return hub.pull(ref).messages[0].prompt.template


def ensure_prompt(hub) -> str:
    """Publish v1 as "prod" the first time, so lessons and the improvement loop share one prompt history."""
    try:
        return prompt_text(hub)
    except KeyError:
        hub.push(PROMPT_NAME, prompt_template(SUPPORT_PROMPT_V1), tags=["prod"], description="v1: the launch prompt")
        return SUPPORT_PROMPT_V1


def load_conversations(path: Path = TRAFFIC_PATH, limit: int | None = None) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))[:limit]


def simulate_traffic(
    agent,
    conversations: list[dict],
    *,
    path: Path = TRACE_LOG,
    variant: str = "prod",
    tags: Iterable[str] = (),
    fault: Callable[[str], None] | None = None,
) -> list[str]:
    """Play conversations through the agent like real users; each turn is one trace. Returns the trace ids.

    A failing turn is recorded as an error trace, and the next conversation goes on: production keeps serving.
    `fault(question)` runs before each turn and may raise, to inject an incident on purpose (lesson 9.3).
    """
    trace_ids = []
    for conversation in conversations:
        customer = Customer(conversation["name"], conversation["plan"], conversation["account_id"])
        thread = f"{conversation['id']}/{variant}"
        config = {"configurable": {"thread_id": thread}}
        metadata = {"thread_id": thread, "account_id": customer.account_id, "plan": customer.plan, "variant": variant}
        for turn in conversation["turns"]:
            with traced(
                "support_turn", {"question": turn}, metadata=metadata, tags=["production", *tags], path=path
            ) as run:
                try:
                    if fault is not None:
                        fault(turn)
                    result = agent.invoke({"messages": [{"role": "user", "content": turn}]}, config, context=customer)
                    run.end(outputs={"answer": result["messages"][-1].text})
                except Exception as err:  # noqa: BLE001 - recorded on the trace, like any production error
                    run.end(error=f"{type(err).__name__}: {err}")
            trace_ids.append(str(run.trace_id))
    return trace_ids


def ensure_traffic(
    llm: BaseChatModel,
    retriever: BaseRetriever | None = None,
    *,
    path: Path = TRACE_LOG,
    limit: int | None = None,
    system_prompt: str = SUPPORT_PROMPT_V1,
) -> list[dict]:
    """The trace log, generating it first if it's empty, so each lesson works alone and later ones run fast."""
    if not load_rows(path):
        simulate_traffic(
            build_monitored_agent(llm, retriever, system_prompt=system_prompt),
            load_conversations(limit=limit),
            path=path,
        )
    return load_rows(path)


# --- Reading traces: one row per trace, like LangSmith's runs table -----------------------------------------


def summarize(rows: list[dict]) -> list[dict]:
    """Group the trace log's runs by trace into one summary per root run (oldest first)."""
    by_trace: dict[str, list[dict]] = {}
    for row in rows:
        by_trace.setdefault(row["trace_id"], []).append(row)
    summaries = []
    for trace_rows in by_trace.values():
        root = next(r for r in trace_rows if r["parent_id"] is None)
        llm_runs = [r for r in trace_rows if r["run_type"] == "llm"]
        summaries.append(
            {
                "trace_id": root["trace_id"],
                "name": root["name"],
                "run_type": root["run_type"],
                "start": root["start"],
                "status": "error" if any(r["error"] for r in trace_rows) else "success",
                "error": next((r["error"] for r in trace_rows if r["error"]), None),
                "latency": (root["latency_ms"] or 0) / 1000,  # seconds, as in LangSmith filters
                "total_tokens": sum(r["tokens"] or 0 for r in llm_runs),
                "llm_calls": len(llm_runs),
                "tools": [r["name"] for r in trace_rows if r["run_type"] == "tool"],
                # every run's tags, like LangSmith's tree filters: a guard deep inside the trace can flag it
                "tags": list(
                    dict.fromkeys(
                        t
                        for r in [root, *trace_rows]
                        for t in r.get("tags", [])
                        if not t.startswith(("graph:", "seq:"))
                    )
                ),
                "metadata": root.get("metadata", {}),
                "inputs": root["inputs"],
                "outputs": root["outputs"],
                "runs": trace_rows,
            }
        )
    return sorted(summaries, key=lambda s: s["start"])


def question_of(summary: dict) -> str:
    return json.loads(summary["inputs"]).get("question", "") if summary["inputs"].startswith("{") else summary["inputs"]


def answer_of(summary: dict) -> str:
    try:
        return (json.loads(summary["outputs"]) or {}).get("answer", "")
    except (json.JSONDecodeError, AttributeError):
        return summary["outputs"]


# --- LangSmith's filter language, locally ---------------------------------------------------------------

COMPARE = {
    "eq": lambda a, b: a == b,
    "neq": lambda a, b: a != b,
    "gt": lambda a, b: a is not None and a > b,
    "gte": lambda a, b: a is not None and a >= b,
    "lt": lambda a, b: a is not None and a < b,
    "lte": lambda a, b: a is not None and a <= b,
}


def run_filter(expression: str) -> Callable[[dict], bool]:
    """Compile a LangSmith run filter, e.g. `and(eq(status, "error"), gt(latency, 5))`, into a predicate.

    Supported: and, or, not; eq, neq, gt, gte, lt, lte on `status`, `name`, `run_type`, `latency` (seconds),
    `total_tokens`; `has(tags, "x")`; `search("text")` in inputs and outputs; and metadata as LangSmith writes
    it: `and(eq(metadata_key, "plan"), eq(metadata_value, "team"))`, and feedback the same way:
    `and(eq(feedback_key, "helpfulness"), lt(feedback_score, 1))`. The same string works with
    `Client().list_runs(filter=...)` and in LangSmith's filter bar and automation rules.
    """
    # The syntax is Python's call syntax, except that and/or/not are Python keywords: rename them to parse.
    python = re.sub(r"\b(and|or|not)\s*\(", r"\1_(", expression.strip())
    tree = ast.parse(python, mode="eval").body

    def value(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return {"true": True, "false": False, "null": None}.get(node.id, node.id)
        raise ValueError(f"unsupported value in filter: {ast.dump(node)}")

    def compile_node(node) -> Callable[[dict], bool]:
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            raise ValueError(f"expected a call like eq(field, value), got {ast.unparse(node)}")
        op, args = node.func.id.removesuffix("_"), node.args
        if op in ("and", "or"):
            parts = [compile_node(a) for a in args if not _pair_field(a)]
            if op == "and":
                parts += [_pair_predicate(container, args) for container in PAIRS if _has_pair(container, args)]
            combine = all if op == "and" else any
            return lambda row: combine(p(row) for p in parts)
        if op == "not":
            inner = compile_node(args[0])
            return lambda row: not inner(row)
        if op == "has":
            field_name, wanted = value(args[0]), value(args[1])
            return lambda row: wanted in (row.get(field_name) or [])
        if op == "search":
            needle = str(value(args[0])).lower()
            return lambda row: needle in f"{row.get('inputs', '')} {row.get('outputs', '')}".lower()
        if op in COMPARE:
            field_name, wanted = value(args[0]), value(args[1])
            compare = COMPARE[op]
            return lambda row: compare(row.get(field_name), wanted)
        raise ValueError(f"unknown filter function {op!r}")

    return compile_node(tree)


# Key/value pairs LangSmith filters on: the key's eq() and the value's comparison sit side by side in an and().
PAIRS = {"metadata": ("metadata_key", "metadata_value"), "feedback": ("feedback_key", "feedback_score")}


def _pair_field(node) -> str | None:
    if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Name):
        if any(node.args[0].id in names for names in PAIRS.values()):
            return node.args[0].id
    return None


def _has_pair(container: str, args) -> bool:
    return any(_pair_field(a) == PAIRS[container][0] for a in args)


def _pair_predicate(container: str, args) -> Callable[[dict], bool]:
    """`eq(metadata_key, "plan")` + `eq(metadata_value, "team")` -> row["metadata"]["plan"] == "team"."""
    key_field, value_field = PAIRS[container]
    key = next(a.args[1].value for a in args if _pair_field(a) == key_field)
    compare_node = next((a for a in args if _pair_field(a) == value_field), None)
    if compare_node is None:
        return lambda row: key in (row.get(container) or {})
    compare, wanted = COMPARE[compare_node.func.id], compare_node.args[1].value
    return lambda row: compare((row.get(container) or {}).get(key), wanted)


# --- Cost and latency -------------------------------------------------------------------------------------


def load_pricing(path: Path = PRICING_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def price_for(model: str | None, pricing: dict) -> dict:
    """USD per million input/output tokens for a model name (exact or prefix match); unknown models cost 0."""
    models = pricing.get("models", {})
    if model in models:
        return models[model]
    return next((p for name, p in models.items() if model and model.startswith(name)), {"input": 0.0, "output": 0.0})


def run_cost(row: dict, price: dict) -> float:
    """The cost of one LLM run: tokens in and out, each at its own rate (output tokens usually cost more)."""
    return ((row.get("input_tokens") or 0) * price["input"] + (row.get("output_tokens") or 0) * price["output"]) / 1e6


def trace_cost(summary: dict, pricing: dict, price: dict | None = None) -> float:
    """A trace's cost: its LLM runs at their model's price, or all at `price` (to ask "what if we used X?")."""
    return sum(
        run_cost(r, price or price_for(r.get("model"), pricing)) for r in summary["runs"] if r["run_type"] == "llm"
    )


def percentile(values: list[float], p: float) -> float:
    """The p-th percentile by nearest rank: p95 is the latency 95% of requests beat."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


# --- Online evaluators: judging production traffic, without reference answers -----------------------------


class Sentiment(BaseModel):
    """How the customer feels, judged from their own words."""

    label: Literal["negative", "neutral", "positive"] = Field(description="The customer's mood in this message.")
    reason: str = Field(description="One short sentence quoting what shows it.")


class Helpfulness(BaseModel):
    """Did the reply actually help? (No reference answer: production has none.)"""

    helpful: bool = Field(description="True if the reply answers the question or gives a concrete next step.")
    reason: str = Field(description="One short sentence.")


SENTIMENT_SCORE = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}


def sentiment_evaluator(llm: BaseChatModel) -> Callable[[dict], dict]:
    judge = structured_llm(llm, Sentiment)

    def user_sentiment(summary: dict) -> dict:
        verdict = judge.invoke(f"Classify the sentiment of this customer message:\n\n{question_of(summary)}")
        return {"key": "user_sentiment", "score": SENTIMENT_SCORE[verdict.label], "comment": verdict.reason}

    return user_sentiment


def helpfulness_evaluator(llm: BaseChatModel) -> Callable[[dict], dict]:
    judge = structured_llm(llm, Helpfulness)

    def helpfulness(summary: dict) -> dict:
        verdict = judge.invoke(
            "You review a support assistant's reply.\n\n"
            f"Customer: {question_of(summary)}\n\nReply: {answer_of(summary)}"
        )
        return {"key": "helpfulness", "score": float(verdict.helpful), "comment": verdict.reason}

    return helpfulness


def cites_docs(summary: dict) -> dict:
    """A code evaluator: free, instant, deterministic. Did the answer cite a doc page?"""
    return {"key": "cites_docs", "score": float(".md" in answer_of(summary)), "comment": None}


@dataclass
class OnlineEvaluator:
    """An evaluator rule: which traces (a filter), how many of them (a sampling rate), and how to score."""

    evaluate: Callable[[dict], dict]
    sample_rate: float = 1.0
    filter: str | None = None


def run_online_evaluators(
    summaries: list[dict], evaluators: list[OnlineEvaluator], *, feedback_path: Path = FEEDBACK_LOG, seed: int = 0
) -> list[dict]:
    """Score traces as they arrive and attach each score as feedback on the trace (LangSmith or the local log)."""
    sampler = random.Random(seed)
    records = []
    for evaluator in evaluators:
        selected = run_filter(evaluator.filter) if evaluator.filter else (lambda _row: True)
        for summary in summaries:
            if not selected(summary) or sampler.random() >= evaluator.sample_rate:
                continue
            try:
                result = evaluator.evaluate(summary)
            except Exception as err:  # noqa: BLE001 - a broken judge must not stop monitoring; LangSmith logs these too
                print(f"  (evaluator {evaluator.evaluate.__name__} failed on a trace: {type(err).__name__})")
                continue
            records.append(
                record_feedback(
                    summary["trace_id"], result["key"], result["score"], result["comment"], path=feedback_path
                )
            )
    return records


def default_evaluators(llm: BaseChatModel) -> list[OnlineEvaluator]:
    """The rules lesson 9.4 sets up: sentiment and helpfulness (LLM judges) and doc citations (code)."""
    return [
        OnlineEvaluator(sentiment_evaluator(llm)),
        OnlineEvaluator(helpfulness_evaluator(llm)),
        OnlineEvaluator(cites_docs),
    ]


def ensure_feedback(
    llm: BaseChatModel, summaries: list[dict], *, feedback_path: Path = FEEDBACK_LOG, evaluators=None
) -> list[dict]:
    """Run each evaluator on the traces it hasn't scored yet (as online evaluators do when traces arrive).

    An evaluator's feedback key is its function name (`user_sentiment`, `helpfulness`, `cites_docs`).
    """
    scored = feedback_by_trace(load_rows(feedback_path))
    for evaluator in evaluators or default_evaluators(llm):
        key = evaluator.evaluate.__name__
        new = [s for s in summaries if key not in scored.get(s["trace_id"], {})]
        if new:
            run_online_evaluators(new, [evaluator], feedback_path=feedback_path)
    return load_rows(feedback_path)


def with_feedback(summaries: list[dict], feedback: list[dict]) -> list[dict]:
    """Attach each trace's feedback scores (`summary["feedback"]`), so filters can use them."""
    scores = feedback_by_trace(feedback)
    return [{**s, "feedback": scores.get(s["trace_id"], {})} for s in summaries]


def feedback_by_trace(feedback: list[dict]) -> dict[str, dict[str, float]]:
    """{trace id: {feedback key: score}}: the latest score for each key."""
    scores: dict[str, dict[str, float]] = {}
    for record in feedback:
        scores.setdefault(record["run_id"], {})[record["key"]] = record["score"]
    return scores


# --- Automations and alerts -------------------------------------------------------------------------------


@dataclass
class Rule:
    """An automation: for traces matching `filter` (sampled at `sample_rate`), run each action."""

    name: str
    filter: str
    actions: list[Callable[[dict], str]]
    sample_rate: float = 1.0


def apply_rules(summaries: list[dict], rules: list[Rule], *, seed: int = 0) -> list[dict]:
    sampler = random.Random(seed)
    fired = []
    for rule in rules:
        matches = run_filter(rule.filter)
        for summary in summaries:
            if matches(summary) and sampler.random() < rule.sample_rate:
                fired.append(
                    {"rule": rule.name, "trace_id": summary["trace_id"], "done": [a(summary) for a in rule.actions]}
                )
    return fired


def add_to_dataset(path: Path) -> Callable[[dict], str]:
    """Action: save the trace as a dataset example (inputs, plus its output as a reference to review)."""

    def action(summary: dict) -> str:
        example = {
            "inputs": {"question": question_of(summary)},
            "outputs": {"answer": answer_of(summary)},
            "source_trace_id": summary["trace_id"],
        }
        append_rows(path, [example])
        return f"added to dataset {path.stem}"

    return action


def add_to_annotation_queue(path: Path) -> Callable[[dict], str]:
    """Action: send the trace to people for review (lesson 5.12's queue)."""

    def action(summary: dict) -> str:
        append_rows(
            path, [{"run_id": summary["trace_id"], "question": question_of(summary), "answer": answer_of(summary)}]
        )
        return f"queued for review in {path.stem}"

    return action


def webhook(sink: list) -> Callable[[dict], str]:
    """Action: notify another system. A real rule POSTs JSON to a URL; here the payload goes into `sink`."""

    def action(summary: dict) -> str:
        sink.append({"trace_id": summary["trace_id"], "status": summary["status"], "question": question_of(summary)})
        return "webhook sent"

    return action


def metrics(summaries: list[dict], feedback: list[dict] | None = None, pricing: dict | None = None) -> dict:
    """The numbers a monitoring dashboard shows, for a set of traces."""
    scores = feedback_by_trace(feedback or [])
    sentiments = [
        scores[s["trace_id"]]["user_sentiment"] for s in summaries if "user_sentiment" in scores.get(s["trace_id"], {})
    ]
    latencies = [s["latency"] for s in summaries]
    return {
        "traces": len(summaries),
        "error_rate": sum(s["status"] == "error" for s in summaries) / len(summaries) if summaries else 0.0,
        "p50_latency": percentile(latencies, 50),
        "p95_latency": percentile(latencies, 95),
        "tokens": sum(s["total_tokens"] for s in summaries),
        "cost": sum(trace_cost(s, pricing) for s in summaries) if pricing else 0.0,
        "negative_rate": sum(v < 0 for v in sentiments) / len(sentiments) if sentiments else 0.0,
        "flagged": sum("prompt_injection" in s["tags"] for s in summaries),
    }


@dataclass
class AlertRule:
    """Alert when `metric` (a key of `metrics()`) crosses `threshold` over the last `window_minutes`."""

    name: str
    metric: str
    threshold: float
    op: Literal["above", "below"] = "above"
    window_minutes: int = 15
    min_traces: int = 1  # don't page anyone over one request


def check_alerts(
    summaries: list[dict],
    rules: list[AlertRule],
    *,
    feedback: list[dict] | None = None,
    pricing: dict | None = None,
    now: datetime | None = None,
    path: Path | None = ALERTS_LOG,
) -> list[dict]:
    """Evaluate alert rules over a time window; firing alerts are logged (and would page or post to a webhook)."""
    if not summaries:
        return []
    now = now or max(datetime.fromisoformat(s["start"]) for s in summaries)
    fired = []
    for rule in rules:
        since = now - timedelta(minutes=rule.window_minutes)
        window = [s for s in summaries if datetime.fromisoformat(s["start"]) >= since]
        if len(window) < rule.min_traces:
            continue
        value = metrics(window, feedback, pricing)[rule.metric]
        if (value > rule.threshold) if rule.op == "above" else (value < rule.threshold):
            alert = {
                "alert": rule.name,
                "metric": rule.metric,
                "value": round(value, 4),
                "threshold": rule.threshold,
                "window_minutes": rule.window_minutes,
                "traces": len(window),
                "at": now.isoformat(),
            }
            fired.append(alert)
    if path is not None and fired:
        append_rows(path, fired)
    return fired


# --- A/B tests ---------------------------------------------------------------------------------------------


def assign_variant(key: str, variants: dict[str, float], *, experiment: str = "ab") -> str:
    """Deterministic assignment: the same customer always sees the same variant (`variants`: name -> share)."""
    bucket = int(hashlib.sha256(f"{experiment}:{key}".encode()).hexdigest(), 16) % 10_000 / 10_000
    total = 0.0
    for name, share in variants.items():
        total += share
        if bucket < total:
            return name
    return name


def two_proportion_test(successes_a: int, n_a: int, successes_b: int, n_b: int) -> dict:
    """Is the difference between two success rates real or noise? (Two-proportion z-test, two-sided.)"""
    if not n_a or not n_b:
        return {"rate_a": 0.0, "rate_b": 0.0, "z": 0.0, "p_value": 1.0}
    rate_a, rate_b = successes_a / n_a, successes_b / n_b
    pooled = (successes_a + successes_b) / (n_a + n_b)
    spread = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = (rate_b - rate_a) / spread if spread else 0.0
    p_value = math.erfc(abs(z) / math.sqrt(2))
    return {"rate_a": rate_a, "rate_b": rate_b, "z": round(z, 3), "p_value": round(p_value, 4)}
