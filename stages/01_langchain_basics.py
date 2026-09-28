"""Stage 1 - LangChain basics: models, prompts, chains (LCEL), streaming, structured output, batching.

What LangChain gives you here:
  * one interface for every chat model provider (invoke / stream / batch / with_structured_output)
  * reusable prompt templates
  * the `|` operator (LCEL) to compose prompt -> model -> parser into a single Runnable

Run:  uv run poe stage1        (Docker: docker compose run --rm app poe stage1)
"""

from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from docshelp.config import Settings, describe, ensure_llm_ready, get_llm, get_provider, structured_llm


def header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    ensure_llm_ready()
    llm = get_llm()
    print(f"Using {describe()}  (change with DOCSHELP_PROVIDER / DOCSHELP_MODEL)")

    # 1. Call a model directly. Messages in, an AIMessage out.
    header("1. Direct model call")
    reply = llm.invoke("In one sentence, what is retrieval-augmented generation?")
    print(reply.text)
    print(f"(tokens: {reply.usage_metadata})")

    # 2. A prompt template + model + output parser, composed with LCEL's `|`.
    #    The result `chain` is itself a Runnable with invoke/stream/batch.
    header("2. Prompt template | model | parser  (LCEL chain)")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a support agent for {product}. Reply in at most two sentences."),
            ("human", "{question}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    print(chain.invoke({"product": "Nimbus Notes", "question": "Can I use the app offline?"}))
    print("\nNote: the model has no access to our docs yet, so it is guessing. Stage 2 fixes that.")

    # 3. Streaming: the same chain, token by token.
    header("3. Streaming")
    for token in chain.stream({"product": "Nimbus Notes", "question": "Write a haiku about note-taking."}):
        print(token, end="", flush=True)
    print()

    # 4. Structured output: get a validated Pydantic object instead of free text.
    header("4. Structured output (triage an incoming support email)")

    class Triage(BaseModel):
        category: Literal["billing", "sync", "sharing", "security", "import_export", "other"]
        sentiment: Literal["positive", "neutral", "negative"]
        urgent: bool = Field(description="True if the customer is blocked from working.")
        summary: str = Field(description="One-line summary.")

    triage = structured_llm(llm, Triage)
    email = (
        "Hi, since this morning none of my notes sync between my laptop and phone, the cloud icon "
        "has a red line through it. I have a client presentation in 2 hours. Please help!!"
    )
    result = triage.invoke(email)
    print(repr(result))

    # 5. Batch: run many inputs in parallel with one call.
    header("5. Batch")
    emails = [
        "How do I get an invoice for last month?",
        "Love the new dark mode, thanks team!",
        "Can guests see all our notebooks?",
    ]
    for e, t in zip(emails, triage.batch(emails)):
        print(f"{t.category:14} {t.sentiment:9} urgent={t.urgent!s:5} <- {e}")

    # 6. Swapping models is a one-line change: same chain code, different model.
    #    (Swapping *providers* is just DOCSHELP_PROVIDER in .env; no code changes at all.)
    settings = Settings.from_env()
    fast_model = settings.fast_model or get_provider(settings).fast_model
    header(f"6. Swap the model ({fast_model}, same chain code)")
    fast_chain = prompt | get_llm(fast=True) | StrOutputParser()
    print(fast_chain.invoke({"product": "Nimbus Notes", "question": "What is a notebook?"}))


if __name__ == "__main__":
    main()
