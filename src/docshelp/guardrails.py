"""Security for a production agent: keep PII away from models and traces, and catch prompt injection (lesson 9.6).

Three layers, each one also a monitoring signal:
  * `pii_guards()`: LangChain's prebuilt `PIIMiddleware` redacts emails and masks card numbers in the
    customer's message *before* the model sees it, so they never reach the model, its provider, or the trace
    of the model call.
  * `trace_anonymizer()`: for LangSmith, `Client(anonymizer=...)` scrubs the same patterns from everything the
    client uploads (inputs, outputs, metadata), as a second line of defence.
  * `injection_guard`: a `before_agent` hook that spots common prompt-injection phrasing, answers with a
    refusal without calling the model, and tags its run `prompt_injection`, so monitoring can count attempts
    and alert on spikes (9.5). The tag sits on a run inside the trace, not on the root: LangSmith finds such
    traces with a *tree* filter (`list_runs(is_root=True, tree_filter='has(tags, "prompt_injection")')`), and
    `monitoring.summarize` gives each trace the tags of all its runs for the same reason. Pattern lists catch
    the obvious; production systems add a classifier model, but the monitoring side (tag, count, alert) is
    the same.
"""

from __future__ import annotations

import re

from langchain.agents.middleware import AgentState, PIIMiddleware, before_agent
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime
from langsmith import get_current_run_tree
from langsmith.anonymizer import create_anonymizer

EMAIL = r"[\w.+-]+@[\w-]+\.[\w.-]+"
CARD = r"\b(?:\d[ -]?){13,19}\b"

INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|rules)",
    r"disregard (your|all|the) (guidelines|instructions|rules)",
    r"(print|reveal|show|repeat) (me )?(your|the) (system prompt|instructions)",
    r"\byou are now\b",
    r"\bDAN\b",
    r"api keys?",
]
REFUSAL = "I can't help with that. I can answer questions about Nimbus Notes, your account and billing."


def pii_guards() -> list[PIIMiddleware]:
    return [
        PIIMiddleware("email", strategy="redact"),  # -> [REDACTED_EMAIL]
        PIIMiddleware("credit_card", strategy="mask"),  # -> ****-****-****-1111
    ]


def trace_anonymizer():
    """Pass as `langsmith.Client(anonymizer=trace_anonymizer())` to scrub uploads to LangSmith."""
    return create_anonymizer([{"pattern": EMAIL, "replace": "<email>"}, {"pattern": CARD, "replace": "<card>"}])


def looks_like_injection(text: str) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in INJECTION_PATTERNS)


@before_agent(can_jump_to=["end"])
def injection_guard(state: AgentState, runtime: Runtime) -> dict | None:
    """Refuse and flag likely prompt injections before any model call."""
    last = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if last is None or not looks_like_injection(last.text):
        return None
    run = get_current_run_tree()  # this hook's own run, inside the trace
    if run is not None:  # the tag makes the trace findable: `has(tags, "prompt_injection")` as a tree filter
        run.add_tags(["prompt_injection"])
        run.add_metadata({"security": "prompt_injection_blocked"})
    return {"messages": [AIMessage(REFUSAL)], "jump_to": "end"}
