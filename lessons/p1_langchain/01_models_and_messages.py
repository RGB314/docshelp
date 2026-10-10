"""Models and messages: one interface for every chat model.

Academy: LangChain Essentials L2 "Models and Messages", Intro to LangChain M1 L1 "Foundational Models".

Concepts:
  * a chat model takes messages in and returns an AIMessage (text, token usage, metadata)
  * message types: SystemMessage (instructions), HumanMessage (user), AIMessage (model), and the
    equivalent plain-dict form {"role": ..., "content": ...}
  * the model is stateless: "memory" is just sending the earlier messages again
  * batch() runs many inputs in parallel; swapping models or providers needs no code change

Run:  uv run poe lesson 1.1
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header


def run(llm: BaseChatModel, fast_llm: BaseChatModel) -> dict:
    header("1. A single call: a string in, an AIMessage out")
    reply = llm.invoke("In one sentence, what is retrieval-augmented generation?")
    print(reply.text)
    print(f"(tokens: {reply.usage_metadata})")

    header("2. Message types: system instructions + user message")
    history = [
        SystemMessage("You are DocsHelp, the support assistant for Nimbus Notes. Be brief."),
        HumanMessage("Hi, I'm Ada."),
    ]
    first = llm.invoke(history)
    print(f"DocsHelp: {first.text}")

    header("3. The model is stateless: resend the history to continue the conversation")
    history += [first, HumanMessage("What's my name?")]
    # Dicts work too: {"role": "user", "content": "..."} is the same as HumanMessage("...").
    second = llm.invoke(history)
    print(f"DocsHelp: {second.text}")
    print(f"(sent {len(history)} messages; lesson 1.8 lets a checkpointer keep this history for you)")

    header("4. batch(): many independent inputs, run in parallel")
    questions = ["Name one note-taking feature.", "Name one sync feature.", "Name one sharing feature."]
    batch = [m.text for m in llm.batch(questions)]
    for q, a in zip(questions, batch):
        print(f"- {q} -> {a}")

    header("5. Swap the model: same code, the provider's smaller 'fast' model")
    fast = fast_llm.invoke("What is a notebook in a notes app? One sentence.").text
    print(fast)
    print("(Swapping the *provider* is just DOCSHELP_PROVIDER in .env; no code changes at all.)")

    return {"reply": reply, "history": history + [second], "batch": batch, "fast": fast}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}  (change with DOCSHELP_PROVIDER / DOCSHELP_MODEL)")
    run(get_llm(), get_llm(fast=True))


if __name__ == "__main__":
    main()
