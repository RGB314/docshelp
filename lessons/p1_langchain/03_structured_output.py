"""Structured output: get a validated Pydantic object instead of free text.

Academy: LangChain Essentials L7 "Structured Output".

Concepts:
  * describe the shape you want as a Pydantic model; field descriptions become instructions
  * `structured_llm(llm, Schema)` wraps `llm.with_structured_output(...)` using the method the configured
    provider supports best (JSON schema or tool calling), so the code stays provider-neutral
  * structured runnables batch like any other (triage a whole inbox at once)
  * agents return structured output too: see `response_format` in the capstone (lesson 1.10)

Real-world use: triage incoming support emails into fields your ticketing system can route on.

Run:  uv run poe lesson 1.3
"""

from typing import Literal

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from docshelp.config import describe, ensure_llm_ready, get_llm, structured_llm
from docshelp.console import header


class Triage(BaseModel):
    category: Literal["billing", "sync", "sharing", "security", "import_export", "other"]
    sentiment: Literal["positive", "neutral", "negative"]
    urgent: bool = Field(description="True if the customer is blocked from working.")
    summary: str = Field(description="One-line summary.")


INBOX = [
    "How do I get an invoice for last month?",
    "Love the new dark mode, thanks team!",
    "Can guests see all our notebooks?",
]


def run(llm: BaseChatModel) -> dict:
    triage = structured_llm(llm, Triage)

    header("1. One email -> one validated object")
    email = (
        "Hi, since this morning none of my notes sync between my laptop and phone, the cloud icon "
        "has a red line through it. I have a client presentation in 2 hours. Please help!!"
    )
    single = triage.invoke(email)
    print(repr(single))
    print(f"Route to: {'on-call engineer' if single.urgent else 'normal queue'}")

    header("2. Triage a whole inbox with batch()")
    batch = triage.batch(INBOX)
    for e, t in zip(INBOX, batch):
        print(f"{t.category:14} {t.sentiment:9} urgent={t.urgent!s:5} <- {e}")

    return {"single": single, "batch": batch}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm())


if __name__ == "__main__":
    main()
