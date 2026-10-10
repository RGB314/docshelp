"""Long-term memory for DocsHelp: what the assistant remembers about a customer across conversations.

Short-term memory (the checkpointer, Parts 1–3) lives inside one thread. Long-term memory lives in a
*Store*, shared across threads, organised by namespace, here `(kind, account_id)`:

  ("profile", account)       one document, a `CustomerProfile`, updated in place    (lesson 4.7)
  ("issues", account)        a collection of `PastIssue` items, inserted or updated  (lesson 4.8)
  ("instructions", account)  free text: how this customer likes to be helped

`trustcall` (by the LangChain team) extracts and *patches* these schemas from a conversation, so updates
don't lose fields the conversation didn't mention. `build_memory_agent` is Part 4's capstone (4.9).
"""

from __future__ import annotations

from typing import Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import Checkpointer
from pydantic import BaseModel, Field
from trustcall import create_extractor

from docshelp.customers import Customer
from docshelp.tools import make_search_docs


class CustomerProfile(BaseModel):
    """What we know about the customer. Leave fields empty when unknown."""

    name: str | None = Field(default=None, description="The customer's first name.")
    role: str | None = Field(default=None, description="Their job or role, e.g. 'workspace admin'.")
    operating_systems: list[str] = Field(default_factory=list, description="e.g. Windows, macOS, iOS.")
    devices: list[str] = Field(default_factory=list, description="e.g. laptop, iPhone, iPad.")
    preferences: list[str] = Field(default_factory=list, description="How they like to be helped.")


class PastIssue(BaseModel):
    """A problem the customer reported."""

    summary: str = Field(description="The problem in one sentence.")
    platform: str | None = Field(default=None, description="Where it happens, e.g. Windows.")
    status: Literal["open", "solved"] = "open"
    resolution: str | None = Field(default=None, description="What fixed it, once solved.")


class UpdateMemory(BaseModel):
    """Save something worth remembering about this customer for future conversations."""

    update_type: Literal["profile", "issues", "instructions"] = Field(
        description="profile: facts about the customer. issues: a problem they reported or its outcome. "
        "instructions: how they want to be helped (tone, length, format)."
    )


def ns(kind: str, account_id: str) -> tuple[str, str]:
    return (kind, account_id)


def make_store(embeddings: Embeddings | None = None, fields: list[str] | None = None) -> InMemoryStore:
    """An in-memory store; with `embeddings`, its items can also be searched by meaning (lesson 4.6).

    Production apps use `PostgresStore` (same interface), or the store LangGraph Server provides (Part 8).
    """
    if embeddings is None:
        return InMemoryStore()
    dims = len(embeddings.embed_query("dimension probe"))
    return InMemoryStore(index={"embed": embeddings, "dims": dims, "fields": fields or ["$"]})


def conversation_only(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Human turns and plain AI answers: what an extractor should read (no pending tool calls)."""
    return [m for m in messages if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not m.tool_calls)]


def _extract(extractor, request: dict) -> dict | None:
    """Run a trustcall extractor; on failure keep the old memory instead of failing the user's turn.

    Writing memory is a side job: an answer shouldn't fail because remembering failed. (Small models
    can trip trustcall's repair step, which also has a bug with LangGraph 1.x.)
    """
    try:
        return extractor.invoke(request)
    except Exception as err:  # noqa: BLE001 - best effort by design; the warning says what happened
        print(f"  (memory not updated: {type(err).__name__}: {err})")
        return None


def update_profile(    store: BaseStore, account_id: str, extractor, messages: list[BaseMessage]
) -> CustomerProfile | None:
    existing = store.get(ns("profile", account_id), "profile")
    current = {"CustomerProfile": existing.value} if existing else None
    result = _extract(extractor, {"messages": messages, "existing": current})
    if not result or not result["responses"]:  # the model produced nothing valid: keep what we had
        return None
    profile = result["responses"][0]
    store.put(ns("profile", account_id), "profile", profile.model_dump())
    return profile


def update_issues(store: BaseStore, account_id: str, extractor, messages: list[BaseMessage]) -> list[str]:
    items = store.search(ns("issues", account_id), limit=100)
    existing = [(item.key, "PastIssue", item.value) for item in items]
    result = _extract(extractor, {"messages": messages, "existing": existing or None})
    if not result:
        return []
    keys, inserted = [], 0
    for issue, meta in zip(result["responses"], result["response_metadata"]):
        key = meta.get("json_doc_id")  # set when the model patched an existing item
        if key is None:  # a new item
            inserted += 1
            key = f"issue-{len(existing) + inserted}"
        store.put(ns("issues", account_id), key, issue.model_dump())
        keys.append(key)
    return keys


def recall(store: BaseStore, account_id: str) -> str:
    """Everything remembered about a customer, formatted for a system prompt."""
    profile = store.get(ns("profile", account_id), "profile")
    issues = store.search(ns("issues", account_id), limit=20)
    instructions = store.get(ns("instructions", account_id), "instructions")
    issue_lines = "\n".join(f"- {i.value['summary']} ({i.value['status']})" for i in issues) or "(none)"
    return (
        f"Customer profile: {profile.value if profile else '(none yet)'}\n"
        f"Past issues:\n{issue_lines}\n"
        f"How they like to be helped: {instructions.value['text'] if instructions else '(no preferences yet)'}"
    )


MEMORY_AGENT_PROMPT = (
    "You are DocsHelp, the Nimbus Notes support assistant, and you remember customers across conversations.\n\n"
    "{memory}\n\n"
    "Use search_docs for product questions. Call UpdateMemory when the customer tells you something worth "
    "remembering: facts about them (profile), a problem or its outcome (issues), or how they want to be "
    "helped (instructions). Don't tell them you're saving memories."
)


def build_memory_agent(
    llm: BaseChatModel,
    store: BaseStore,
    retriever: BaseRetriever | None = None,
    checkpointer: Checkpointer | None = None,
):
    """A support agent with long-term memory (Part 4 capstone). Pass `context=Customer(...)` per run."""
    search_docs = make_search_docs(retriever)
    model = llm.bind_tools([UpdateMemory, search_docs])
    profile_extractor = create_extractor(llm, tools=[CustomerProfile], tool_choice="CustomerProfile")
    issue_extractor = create_extractor(llm, tools=[PastIssue], tool_choice="PastIssue", enable_inserts=True)

    def assistant(state: MessagesState, runtime: Runtime[Customer]) -> dict:
        account = (runtime.context or Customer()).account_id
        system = MEMORY_AGENT_PROMPT.format(memory=recall(runtime.store, account))
        return {"messages": [model.invoke([SystemMessage(system), *state["messages"]])]}

    def act(state: MessagesState, runtime: Runtime[Customer]) -> dict:
        """Run every requested tool call: memory updates go to the store, searches to the docs."""
        account = (runtime.context or Customer()).account_id
        conversation = [
            SystemMessage("Extract what's worth remembering from this support conversation."),
            *conversation_only(state["messages"]),
        ]
        results = []
        for call in state["messages"][-1].tool_calls:
            if call["name"] == "search_docs":
                result = search_docs.invoke(call["args"])
            elif call["args"].get("update_type") == "profile":
                result = f"Profile saved: {update_profile(runtime.store, account, profile_extractor, conversation)}"
            elif call["args"].get("update_type") == "issues":
                result = f"Issues saved: {update_issues(runtime.store, account, issue_extractor, conversation)}"
            else:
                ask = HumanMessage("In one sentence, how does this customer want to be helped (tone, length, format)?")
                text = llm.invoke([*conversation, ask]).text.strip()
                runtime.store.put(ns("instructions", account), "instructions", {"text": text})
                result = f"Instructions saved: {text}"
            results.append(ToolMessage(result, tool_call_id=call["id"], name=call["name"]))
        return {"messages": results}

    def route(state: MessagesState) -> Literal["act", "__end__"]:
        return "act" if state["messages"][-1].tool_calls else END

    builder = StateGraph(MessagesState, context_schema=Customer)
    builder.add_node(assistant)
    builder.add_node(act)
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", route)
    builder.add_edge("act", "assistant")
    return builder.compile(checkpointer=checkpointer or InMemorySaver(), store=store, name="memory_agent")
