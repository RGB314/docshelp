"""The ambient support-inbox agent (Part 6.1–6.4): it reacts to incoming emails, not to chat messages.

        email ──▶ triage ──respond──▶ respond ⇄ act ──▶ END
                    │ notify ──▶ triage_review (HITL) ──▶ respond / END
                    └ ignore ──▶ END

  * triage: structured output decides respond / notify / ignore, guided by triage preferences
  * respond + act: a hand-built tool loop (lesson 3.3). It searches the docs and changelog, quotes
    prices, opens tickets and sends the reply, then calls `done`.
  * hitl=True: sending, ticketing and questions pause with an Agent Inbox interrupt (accept / edit /
    ignore / respond), and "notify" emails ask the human whether to reply
  * memory=True: triage and response preferences live in the Store, and every human correction
    (an edit, an ignore, written feedback) updates them, so the agent learns the support lead's style

`HumanInterrupt` / `HumanResponse` below follow the open-source Agent Inbox protocol
(https://github.com/langchain-ai/agent-inbox), so the same graph can be run from that UI (Part 8).
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from typing import Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer, Command, interrupt
from pydantic import BaseModel, Field

from docshelp import tools as support_tools
from docshelp.config import DATA_DIR, structured_llm
from docshelp.customers import ACCOUNTS
from docshelp.sources import search_changelog

INBOX_DIR = DATA_DIR / "inbox"
Triage = Literal["respond", "notify", "ignore"]
PREFS_NAMESPACE = ("inbox_preferences",)
MAX_TOOL_ROUNDS = 8  # per email
MAX_REVIEWS = 6  # requests per email before process_inbox stops asking
DEFAULT_PREFERENCES = {
    "triage": (
        "respond: customers asking questions or reporting problems with Nimbus Notes.\n"
        "notify: automated billing or security alerts, and legal or privacy requests a human must handle.\n"
        "ignore: newsletters, marketing and automatic replies."
    ),
    "response": "Friendly and concise. Sign as 'Nimbus Notes Support'. Mention the doc or release you relied on.",
}


# --- the Agent Inbox protocol -----------------------------------------------------------------------


class ActionRequest(TypedDict):
    action: str  # what the agent wants to do, e.g. "send_reply"
    args: dict  # its arguments, editable by the human


class HumanInterruptConfig(TypedDict):
    allow_ignore: bool  # skip it
    allow_respond: bool  # reply with feedback in words
    allow_edit: bool  # change the arguments
    allow_accept: bool  # run it as proposed


class HumanInterrupt(TypedDict):
    action_request: ActionRequest
    config: HumanInterruptConfig
    description: str  # Markdown shown to the reviewer


class HumanResponse(TypedDict):
    type: Literal["accept", "ignore", "response", "edit"]
    args: Any  # None, feedback text, or an ActionRequest with edited args


REVIEW = {  # which actions need a human, and what the human may do with each
    "send_reply": HumanInterruptConfig(allow_ignore=True, allow_respond=True, allow_edit=True, allow_accept=True),
    "open_ticket": HumanInterruptConfig(allow_ignore=True, allow_respond=True, allow_edit=True, allow_accept=True),
    "ask_human": HumanInterruptConfig(allow_ignore=True, allow_respond=True, allow_edit=False, allow_accept=False),
}


# --- the inbox --------------------------------------------------------------------------------------


def load_inbox() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(INBOX_DIR.glob("*.json"))]


def public(email: dict) -> dict:
    """The email as the agent sees it: without the evaluation labels."""
    return {k: v for k, v in email.items() if k != "eval"}


def format_email(email: dict) -> str:
    return f"**From:** {email['from']}\n**Subject:** {email['subject']}\n\n{email['body']}"


def account_for(sender: str) -> str | None:
    return next((aid for aid, account in ACCOUNTS.items() if sender.lower() in account.contacts), None)


# --- schemas and tools ------------------------------------------------------------------------------


class RouterSchema(BaseModel):
    reasoning: str = Field(description="One sentence on why.")
    classification: Triage = Field(description="respond, notify or ignore, following the preferences.")


class UpdatedPreferences(BaseModel):
    reasoning: str = Field(description="What the human's correction tells you, in one sentence.")
    preferences: str = Field(description="The full, updated preferences. Keep everything still valid.")


@tool("search_changelog")
def search_changelog_tool(query: str) -> str:
    """Search the Nimbus Notes release notes: known issues, fixes and new features by version."""
    return "\n".join(search_changelog(query)) or "Nothing relevant in the changelog."


@tool
def open_ticket(summary: str, priority: Literal["low", "normal", "high", "urgent"]) -> str:
    """Open a ticket for the engineering team: bugs, outages, data loss or account problems."""
    return "(executed by the act node, which knows the sender's account)"


@tool
def ask_human(question: str) -> str:
    """Ask the support lead a question when the docs don't give you the answer."""
    return "(answered by the human through an interrupt)"


@tool
def done() -> str:
    """Call this once the email has been fully handled (reply sent, ticket opened if needed)."""
    return "Done."


def get_preferences(store: BaseStore | None, kind: str) -> str:
    item = store.get(PREFS_NAMESPACE, kind) if store else None
    return item.value["text"] if item else DEFAULT_PREFERENCES[kind]


def update_preferences(llm: BaseChatModel, store: BaseStore, kind: str, evidence: str) -> str:
    """Rewrite one kind of preferences in the light of a human correction, and save it."""
    current = get_preferences(store, kind)
    update = structured_llm(llm, UpdatedPreferences).invoke(
        f"These are the support lead's {kind} preferences:\n{current}\n\n"
        f"The support lead just corrected the assistant:\n{evidence}\n\n"
        "Update the preferences so the assistant gets it right next time. Change only what the correction implies."
    )
    store.put(PREFS_NAMESPACE, kind, {"text": update.preferences})
    return update.preferences


# --- the graph --------------------------------------------------------------------------------------


class InboxState(MessagesState):
    email: dict
    account_id: str | None
    classification: Triage


def classify_email(llm: BaseChatModel, email: dict, preferences: str = DEFAULT_PREFERENCES["triage"]) -> RouterSchema:
    """Triage on its own, so it can be evaluated on its own (lesson 6.2)."""
    return structured_llm(llm, RouterSchema).invoke(
        "You triage the Nimbus Notes support inbox. Classify the email using these preferences:\n"
        f"{preferences}\n\n{format_email(email)}"
    )


def build_inbox_agent(
    llm: BaseChatModel,
    retriever: BaseRetriever | None = None,
    *,
    hitl: bool = False,
    memory: bool = False,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
):
    research_tools = (support_tools.make_search_docs(retriever), search_changelog_tool, support_tools.quote_price)
    research = {t.name: t for t in research_tools}
    actions = [support_tools.send_reply, open_ticket, done, *([ask_human] if hitl else [])]
    model = llm.bind_tools([*research.values(), *actions])

    def prefs(runtime: Runtime, kind: str) -> str:
        return get_preferences(runtime.store if memory else None, kind)

    def learn(runtime: Runtime, kind: str, evidence: str) -> None:
        if memory and runtime.store is not None:
            update_preferences(llm, runtime.store, kind, evidence)

    def respond_message(email: dict, guidance: str = "") -> HumanMessage:
        extra = f"\n\nThe support lead says: {guidance}" if guidance else ""
        return HumanMessage(f"Handle this support email and reply to the sender.{extra}\n\n{format_email(email)}")

    def triage(state: InboxState, runtime: Runtime) -> Command[Literal["respond", "triage_review", "__end__"]]:
        decision = classify_email(llm, state["email"], prefs(runtime, "triage"))
        update = {"classification": decision.classification, "account_id": account_for(state["email"]["from"])}
        if decision.classification == "respond":
            return Command(goto="respond", update={**update, "messages": [respond_message(state["email"])]})
        if decision.classification == "notify" and hitl:
            return Command(goto="triage_review", update=update)
        return Command(goto=END, update=update)

    def triage_review(state: InboxState, runtime: Runtime) -> Command[Literal["respond", "__end__"]]:
        request = HumanInterrupt(
            action_request=ActionRequest(action="Email needs your attention (notify)", args={}),
            config=HumanInterruptConfig(allow_ignore=True, allow_respond=True, allow_edit=False, allow_accept=False),
            description=format_email(state["email"]),
        )
        response: HumanResponse = interrupt([request])[0]
        subject = state["email"]["subject"]
        if response["type"] == "response":  # the human wants a reply after all
            learn(runtime, "triage", f"An email classified 'notify' ({subject!r}) should have been answered.")
            return Command(goto="respond", update={"messages": [respond_message(state["email"], response["args"])]})
        learn(runtime, "triage", f"An email classified 'notify' ({subject!r}) was not worth the human's attention.")
        return Command(goto=END)

    def respond(state: InboxState, runtime: Runtime) -> dict:
        system = (
            "You are the Nimbus Notes support assistant handling the support inbox. Research before answering "
            "(search_docs, search_changelog, quote_price for prices), open a ticket for bugs, outages or data loss, "
            "send exactly one reply with send_reply to the sender, then call done.\n"
            f"Reply style preferences:\n{prefs(runtime, 'response')}"
        )
        return {"messages": [model.invoke([SystemMessage(system), *state["messages"]])]}

    def route(state: InboxState) -> Literal["act", "__end__"]:
        rounds = sum(1 for m in state["messages"] if isinstance(m, AIMessage) and m.tool_calls)
        if rounds > MAX_TOOL_ROUNDS:  # a model that never calls done can't loop forever
            return END
        return "act" if state["messages"][-1].tool_calls else END

    def execute(name: str, args: dict, state: InboxState) -> str:
        if name == "send_reply":
            return support_tools.send_reply.invoke(args)
        if name == "open_ticket":
            ticket = {**args, "account_id": state.get("account_id"), "source": f"inbox:{state['email']['id']}"}
            return f"Opened ticket {support_tools.save_ticket(ticket)}."
        if name == "done":
            return "Done."
        return research[name].invoke(args)

    def act(state: InboxState, runtime: Runtime) -> Command[Literal["respond", "__end__"]]:
        results, finished = [], False
        for call in state["messages"][-1].tool_calls:
            name, args = call["name"], call["args"]
            if hitl and name in REVIEW:
                request = HumanInterrupt(
                    action_request=ActionRequest(action=name, args=args),
                    config=REVIEW[name],
                    description=(
                        f"{format_email(state['email'])}\n\n---\n**Proposed {name}:** {json.dumps(args, indent=2)}"
                    ),
                )
                response: HumanResponse = interrupt([request])[0]
                if response["type"] == "ignore":
                    if name == "send_reply":
                        learn(runtime, "triage", f"The human chose not to answer {state['email']['subject']!r} at all.")
                    results.append(ToolMessage(f"The human ignored this {name}; stop.", tool_call_id=call["id"]))
                    finished = True
                    continue
                if response["type"] == "response":
                    if name == "send_reply":
                        learn(runtime, "response", f"Feedback on a drafted reply: {response['args']}")
                    feedback = f"Human feedback, not executed: {response['args']}"
                    results.append(ToolMessage(feedback, tool_call_id=call["id"]))
                    continue
                if response["type"] == "edit":
                    edited = response["args"]["args"]
                    if name == "send_reply":
                        change = f"Draft:\n{args.get('body')}\n\nThe human changed it to:\n{edited.get('body')}"
                        learn(runtime, "response", change)
                    args = edited
            results.append(ToolMessage(execute(name, args, state), tool_call_id=call["id"], name=name))
            finished = finished or name == "done"
        return Command(goto=END if finished else "respond", update={"messages": results})

    builder = StateGraph(InboxState)
    builder.add_node(triage)
    builder.add_node(respond)
    builder.add_node(act)
    builder.add_node(triage_review)  # only reached when hitl=True (triage's Command type names it)
    builder.add_edge(START, "triage")
    builder.add_conditional_edges("respond", route)
    if hitl and checkpointer is None:
        checkpointer = InMemorySaver()  # interrupts need somewhere to save the paused run
    return builder.compile(checkpointer=checkpointer, store=store, name="inbox_agent")


def tools_called(messages: list) -> list[str]:
    return [c["name"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]


Reviewer = Callable[[str, HumanInterrupt], HumanResponse]  # (email id, request) -> the human's response


def process_inbox(agent, emails: list[dict], reviewer: Reviewer) -> dict[str, dict]:
    """Run each email through a hitl agent, answering every Agent Inbox request with `reviewer`.

    This plays the part of the Agent Inbox UI: show each paused request to a person, send back their response.
    """
    results = {}
    for email in emails:
        print(f"\n{email['id']}  {email['subject']}")
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        state = agent.invoke({"email": public(email)}, config)
        requests = []
        while "__interrupt__" in state:
            request = state["__interrupt__"][0].value[0]  # Agent Inbox sends a list with one request
            requests.append(request["action_request"]["action"])
            decision = reviewer(email["id"], request)
            if len(requests) > MAX_REVIEWS and request["config"]["allow_ignore"]:  # the agent keeps asking: stop
                decision = {"type": "ignore", "args": None}
            note = f" ({decision['args']})" if isinstance(decision["args"], str) else ""
            print(f"  ⏸ {request['action_request']['action']} -> {decision['type']}{note}")
            state = agent.invoke(Command(resume=[decision]), config)
        results[email["id"]] = {
            "classification": state["classification"],
            "requests": requests,
            "tools": tools_called(state.get("messages", [])),
        }
        print(f"  triage={state['classification']}  tools={results[email['id']]['tools']}")
    return results
