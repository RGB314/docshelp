"""Runtime context vs agent state: what the app knows vs what the conversation learns.

Academy: Intro to LangChain M2 L2 "Context and State".

Concepts:
  * runtime context (`context_schema`, read-only for the run): who is calling. `my_account` takes no
    arguments; it reads `runtime.context.account_id`, so the model can't look up someone else's account.
  * `ToolRuntime`: an injected tool parameter that the model never sees, with `.context`, `.state`,
    `.store`, `.tool_call_id` and `.stream_writer`
  * custom state (`state_schema` extending `AgentState`): facts learned *during* the conversation,
    checkpointed with the messages. Here: the customer's device.
  * a tool writes state by returning `Command(update=...)`, which must include the `ToolMessage`
  * another tool reads it back through `runtime.state`, and the final state is returned to the app

Run:  uv run poe lesson 2.2
"""

from typing import Literal, NotRequired

from langchain.agents import AgentState, create_agent
from langchain.tools import ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.customers import Customer
from docshelp.tools import my_account

Device = Literal["windows", "macos", "linux", "ios", "android", "web"]

SYNC_FIXES: dict[str, str] = {
    "windows": "On Windows: right-click the Nimbus tray icon > Sync now. If it stays stuck, sign out and back in.",
    "macos": "On macOS: Nimbus menu-bar icon > Sync now, then check System Settings > Network for a VPN.",
    "linux": "On Linux: run the app with --resync, and make sure the system clock is correct.",
    "ios": "On iOS: pull down on the notes list to refresh, and allow Background App Refresh for Nimbus.",
    "android": "On Android: pull down to refresh, and turn off battery optimisation for Nimbus.",
    "web": "On the web: reload the page; offline edits sync from the desktop or mobile app only.",
}


class SupportState(AgentState):
    """The default agent state (messages, ...) plus what we learn about the customer's setup."""

    device: NotRequired[Device]


@tool
def set_device(device: Device, runtime: ToolRuntime[Customer]) -> Command:
    """Record which device the customer is using, once they've said so."""
    return Command(
        update={
            "device": device,
            "messages": [ToolMessage(f"Noted: the customer uses {device}.", tool_call_id=runtime.tool_call_id)],
        }
    )


@tool
def sync_fix(runtime: ToolRuntime[Customer]) -> str:
    """Device-specific steps to fix sync problems. Call set_device first."""
    device = runtime.state.get("device")
    return SYNC_FIXES[device] if device else "Unknown device: ask the customer which device they use."


def run(llm: BaseChatModel, customer: Customer | None = None) -> dict:
    customer = customer or Customer(name="Ada", plan="team", account_id="acct-1001")
    agent = create_agent(
        llm,
        tools=[my_account, set_device, sync_fix],
        system_prompt=(
            "You are DocsHelp, the Nimbus Notes support assistant. Check the customer's account first. When "
            "they mention their device, record it with set_device, then use sync_fix for troubleshooting steps."
        ),
        state_schema=SupportState,
        context_schema=Customer,
    )
    question = "Hi, notes added on my Windows laptop aren't syncing to my team. Are we over our seat limit?"
    header(f"{customer.name} (context: account {customer.account_id})")
    print(f"User: {question}")
    state = agent.invoke({"messages": [{"role": "user", "content": question}]}, context=customer)
    for message in state["messages"]:
        if isinstance(message, ToolMessage):
            print(f"  [{message.name}] {message.text}")
    answer = state["messages"][-1].text
    print(f"DocsHelp: {answer}")
    header("Final state (returned to the app, and checkpointed if the agent has a checkpointer)")
    print(f"device = {state.get('device', '(not recorded)')}, messages = {len(state['messages'])}")
    return {"state": state, "answer": answer}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm())


if __name__ == "__main__":
    main()
