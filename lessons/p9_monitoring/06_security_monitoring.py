"""Security monitoring: keep PII out of models and traces, block prompt injection, and count the attempts.

Academy: Monitoring Production Agents M4 "Monitoring Security".

Concepts (the pieces are in `docshelp/guardrails.py`):
  * PII: customers paste emails and card numbers. `PIIMiddleware` (prebuilt in LangChain) redacts or masks
    them in the message *before* the model sees it, so neither the model provider nor the trace of the model
    call gets them. Strategies: redact, mask, hash, or block the request.
  * traces are data too: `langsmith.Client(anonymizer=trace_anonymizer())` scrubs the same patterns from what
    is uploaded to LangSmith; `hide_inputs` / `hide_outputs` drop fields entirely.
  * prompt injection: "ignore previous instructions...". `injection_guard` refuses before any model call and
    tags the trace `prompt_injection`. The tag is what makes it *monitorable*: filter `has(tags,
    "prompt_injection")`, a rule that notifies security (9.5), an alert on any attempt, a dashboard tile (9.9).
  * proof, not hope: a callback records exactly what the model received, so the lesson can show the card
    number never reached it.

Run:  uv run poe lesson 9.6
"""

from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.guardrails import injection_guard, pii_guards, trace_anonymizer
from docshelp.ingest import get_retriever
from docshelp.monitoring import (
    TRACE_LOG,
    AlertRule,
    build_monitored_agent,
    check_alerts,
    load_conversations,
    question_of,
    run_filter,
    simulate_traffic,
    summarize,
)
from docshelp.observability import load_rows

SECURITY_CASES = ("conv-04", "conv-05", "conv-11", "conv-01")  # two injections, pasted PII, a normal question


class ModelInputs(BaseCallbackHandler):
    """Records the text of every message sent to a chat model: what the provider actually receives."""

    def __init__(self):
        self.texts: list[str] = []

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.texts += [m.text for batch in messages for m in batch]


def run(llm: BaseChatModel, retriever: BaseRetriever | None = None, *, path: Path = TRACE_LOG) -> dict:
    seen = ModelInputs()
    watched = llm.model_copy(update={"callbacks": [seen]})  # the same model, recording what it receives
    guarded = build_monitored_agent(watched, retriever, middleware=[injection_guard, *pii_guards()])
    cases = [c for c in load_conversations() if c["id"] in SECURITY_CASES]
    header("Security-relevant traffic through the guarded agent")
    new = set(simulate_traffic(guarded, cases, path=path, variant="guarded", tags=["guarded"]))
    summaries = [s for s in summarize(load_rows(path)) if s["trace_id"] in new]
    for s in summaries:
        flag = "BLOCKED" if "prompt_injection" in s["tags"] else "served "
        print(f"  {flag} {s['llm_calls']} model calls  {question_of(s)[:70]}")

    header("What the model received")
    received = "\n".join(seen.texts)
    card_leaked = "4111 1111 1111 1111" in received
    email_leaked = "linus@acme.example" in received
    print(f"  card number reached the model: {card_leaked}; email reached the model: {email_leaked}")
    for text in seen.texts:
        if "charged twice" in text:
            print(f"  the model saw: {text[:110]}")

    header("Scrubbing traces before they leave: the LangSmith anonymizer")
    scrub = trace_anonymizer()
    print(f"  {scrub({'question': 'Card 4111 1111 1111 1111, mail linus@acme.example'})}")

    header("Monitoring it")
    flagged = [s for s in summaries if run_filter('has(tags, "prompt_injection")')(s)]
    alerts = check_alerts(summaries, [AlertRule("Prompt injection attempts", "flagged", 0)], path=None)
    print(f"  {len(flagged)} traces tagged prompt_injection; alerts: {[a['alert'] for a in alerts] or 'none'}")
    return {
        "blocked": [question_of(s) for s in flagged],
        "card_leaked": card_leaked,
        "email_leaked": email_leaked,
        "alerts": [a["alert"] for a in alerts],
        "served": sum(s["llm_calls"] > 0 for s in summaries),
    }


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
