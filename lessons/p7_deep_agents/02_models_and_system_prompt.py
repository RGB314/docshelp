"""Models and the system prompt: which model does what, and what the model actually reads.

Academy: Introduction to Deep Agents M1 "Models", "The System Prompt".

Concepts:
  * models: pass a chat model object (what DocsHelp does, so the provider stays configurable), or a
    `"<provider>:<model>"` string that `init_chat_model` resolves. Each subagent can have its own `model`:
    routine work goes to a smaller, faster model (`get_llm(fast=True)`), and judgment stays with the main one.
  * harness profiles: deep agents adjust themselves per model with a `HarnessProfile` (extra prompt text,
    tool descriptions rewritten for that model, tools left out). Some come built in; you can add your own with
    `register_harness_profile("<provider>", HarnessProfile(...))`.
  * the system prompt is assembled, not just written: yours comes first, then the profile's base prompt and
    suffix, and then each middleware appends its own instructions (here the `write_todos` guide). Read the real
    prompt before tuning it; this lesson prints it with a small `wrap_model_call` middleware (lesson 2.4).
  * keep your prompt about the job (who, what, output format); the harness already explains its tools.

Run:  uv run poe lesson 7.2
"""

from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.deep import budget, deep_agent, list_tickets

PROMPT = "You are the Nimbus Notes Support Ops lead's assistant. Answer in at most three lines."
QUESTION = "Ask the ticket-summarizer for a one-line summary of ticket NIM-7A8B9C, then tell me what it says."


class ShowSystemPrompt(AgentMiddleware):
    """Record the system prompt of every model call: what the model reads, after all the assembly."""

    def __init__(self):
        super().__init__()
        self.prompts: list[str] = []

    def wrap_model_call(self, request, handler):
        self.prompts.append(request.system_message.text if request.system_message else "")
        return handler(request)


def run(llm: BaseChatModel, fast_llm: BaseChatModel | None = None) -> dict:
    summarizer = {
        "name": "ticket-summarizer",
        "description": "Summarises one support ticket in a single line. Give it the ticket id.",
        "system_prompt": "List the tickets, find the one you were given, and summarise it in one line.",
        "tools": [list_tickets],
        "model": fast_llm or llm,  # a smaller, cheaper model is enough for this
        "middleware": [budget()],
    }
    shown = ShowSystemPrompt()
    agent = deep_agent(llm, subagents=[summarizer], planning=True, system_prompt=PROMPT, middleware=[shown])
    result = agent.invoke({"messages": [{"role": "user", "content": QUESTION}]})

    header("The system prompt the main model read (first call)")
    print(shown.prompts[0])
    header("Answer")
    print(result["messages"][-1].text)
    return {"system_prompt": shown.prompts[0], "answer": result["messages"][-1].text}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()} (the subagent uses the fast model)")
    run(get_llm(), get_llm(fast=True))


if __name__ == "__main__":
    main()
