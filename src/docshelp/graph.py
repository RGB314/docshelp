"""The DocsHelp agent as an explicit LangGraph state machine.

                      ┌────────────┐
        START ───────▶│  classify  │  (structured output: intent + standalone question)
                      └─────┬──────┘
          ┌─────────────────┼──────────────────────┐
     docs_question     small_talk            ticket_request
          ▼                 ▼                       ▼
     ┌──────────┐     ┌────────────┐         ┌──────────────┐
  ┌─▶│ retrieve │     │ small_talk │         │ draft_ticket │
  │  └────┬─────┘     └─────┬──────┘         └──────┬───────┘
  │       ▼                 │                       ▼
  │  ┌──────────┐           │                ┌──────────────┐  interrupt(): pauses the run until
  │  │  grade   │           │                │ human_review │  a human approves / edits / rejects
  │  └────┬─────┘           │                └──────┬───────┘
  │  not relevant           │              approve  │  reject
  │  (retry once)           │                ▼      └──────────▶ END
  │  ┌──────────┐           │          ┌─────────────┐
  └──│ rewrite  │           │          │ open_ticket │
     └──────────┘           │          └──────┬──────┘
          relevant / out of retries           │
          ▼                 │                 │
     ┌──────────┐           │                 │
     │ generate │           │                 │
     └────┬─────┘           │                 │
          ▼                 ▼                 ▼
                           END

What LangGraph adds over a plain LangChain chain:
  * explicit, inspectable control flow: branches and loops (grade → rewrite → retrieve)
  * shared, typed state that every node reads and updates
  * persistence: with a checkpointer, every step is saved per `thread_id`, which gives
    multi-turn memory, pause/resume (human-in-the-loop) and time travel for free
"""

from __future__ import annotations

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Checkpointer, Command, interrupt
from pydantic import BaseModel, Field

from docshelp import tools
from docshelp.providers import LLMProvider

MAX_REWRITES = 1


# ---------------------------------------------------------------------------
# State and structured-output schemas
# ---------------------------------------------------------------------------


class State(MessagesState):
    """`messages` (inherited) accumulates the conversation; the rest is per-turn working memory."""

    intent: str
    question: str
    documents: list[dict]
    docs_relevant: bool
    rewrites: int
    ticket: dict | None


class Route(BaseModel):
    intent: Literal["docs_question", "ticket_request", "small_talk"] = Field(
        description="docs_question: anything answerable from product docs. "
        "ticket_request: the user explicitly wants a support ticket / human help. "
        "small_talk: greetings, thanks, off-topic."
    )
    standalone_question: str = Field(
        description="The user's latest message rewritten as a self-contained question, "
        "resolving references to earlier turns."
    )


class Grade(BaseModel):
    relevant: bool = Field(description="True if the documents contain the answer to the question.")
    reason: str


class TicketDraft(BaseModel):
    title: str = Field(description="Short summary, max 10 words.")
    description: str = Field(description="What the user reported and what they already tried.")
    priority: Literal["low", "normal", "high"]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ROUTER_PROMPT = (
    "You route messages for the Nimbus Notes support assistant. Read the conversation and classify "
    "the user's latest message."
)
GRADER_PROMPT = (
    "Decide whether the retrieved documentation excerpts contain the information needed to answer "
    "the question. Partial but sufficient information counts as relevant."
)
REWRITE_PROMPT = (
    "The search query below did not retrieve useful Nimbus Notes documentation. Rewrite it as a "
    "better search query using product terminology. Reply with the query only."
)
ANSWER_PROMPT = (
    "You are DocsHelp, the support assistant for Nimbus Notes. Answer the user's question using only "
    "the documentation excerpts below. Be concise. Cite the source file names in square brackets, "
    "e.g. [07-troubleshooting.md]. If the excerpts don't contain the answer, say you don't know and "
    "offer to open a support ticket.\n\nDocumentation:\n{context}"
)
SMALL_TALK_PROMPT = (
    "You are DocsHelp, the friendly support assistant for Nimbus Notes. Reply briefly. If the user "
    "is off-topic, steer them back to Nimbus Notes questions."
)
TICKET_PROMPT = "Draft a support ticket from this conversation for the Nimbus Notes support team."


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------


def build_graph(
    llm: BaseChatModel | None = None,
    retriever: BaseRetriever | None = None,
    checkpointer: Checkpointer | None = None,
    provider: LLMProvider | None = None,
):
    """Build and compile the graph. Dependencies are injectable so tests can run offline."""
    from docshelp.config import get_llm, get_provider

    provider = provider or get_provider()
    if llm is None:
        llm = get_llm()
    if retriever is None:
        from docshelp.ingest import get_retriever

        retriever = get_retriever()

    def structured(schema):
        # The provider picks the structured-output method its models support best.
        return provider.structured(llm, schema)

    # --- nodes: each takes the current state and returns a partial state update -------------

    def classify(state: State) -> dict:
        route = structured(Route).invoke([SystemMessage(ROUTER_PROMPT), *state["messages"]])
        return {
            "intent": route.intent,
            "question": route.standalone_question,
            "documents": [],
            "rewrites": 0,
            "ticket": None,
        }

    def retrieve(state: State) -> dict:
        docs = retriever.invoke(state["question"])
        return {
            "documents": [
                {"source": d.metadata.get("source", "?"), "content": d.page_content} for d in docs
            ]
        }

    def grade(state: State) -> dict:
        grade = structured(Grade).invoke(
            [
                SystemMessage(GRADER_PROMPT),
                HumanMessage(f"Question: {state['question']}\n\nDocuments:\n{_context(state)}"),
            ]
        )
        return {"docs_relevant": grade.relevant}

    def rewrite(state: State) -> dict:
        new_query = llm.invoke(
            [SystemMessage(REWRITE_PROMPT), HumanMessage(state["question"])]
        ).text.strip()
        return {"question": new_query, "rewrites": state["rewrites"] + 1}

    def generate(state: State) -> dict:
        answer = llm.invoke(
            [SystemMessage(ANSWER_PROMPT.format(context=_context(state))), *state["messages"]]
        )
        return {"messages": [AIMessage(answer.text)]}

    def small_talk(state: State) -> dict:
        reply = llm.invoke([SystemMessage(SMALL_TALK_PROMPT), *state["messages"]])
        return {"messages": [AIMessage(reply.text)]}

    def draft_ticket(state: State) -> dict:
        draft = structured(TicketDraft).invoke([SystemMessage(TICKET_PROMPT), *state["messages"]])
        return {"ticket": draft.model_dump()}

    def human_review(state: State) -> Command[Literal["open_ticket", "__end__"]]:
        # interrupt() saves the state via the checkpointer and stops the run. The value passed
        # here is surfaced to the caller; the run resumes when the caller sends
        # Command(resume=<decision>), and interrupt() then returns that decision.
        decision = interrupt(
            {"question": "Open this support ticket?", "ticket": state["ticket"]}
        )
        action = decision.get("action") if isinstance(decision, dict) else decision
        if action == "approve":
            return Command(goto="open_ticket")
        if action == "edit":
            return Command(goto="open_ticket", update={"ticket": decision["ticket"]})
        return Command(
            goto=END,
            update={"messages": [AIMessage("OK, I won't open a ticket. Anything else I can help with?")]},
        )

    def open_ticket(state: State) -> dict:
        ticket_id = tools.save_ticket(state["ticket"])
        return {
            "messages": [
                AIMessage(
                    f"I've opened ticket {ticket_id}: \"{state['ticket']['title']}\". "
                    "Team and Enterprise customers get a first response within 8 business hours."
                )
            ]
        }

    # --- edges: how control flows between nodes -----------------------------------------------

    def route_intent(state: State) -> str:
        return {"docs_question": "retrieve", "ticket_request": "draft_ticket"}.get(
            state["intent"], "small_talk"
        )

    def after_grade(state: State) -> str:
        if state["docs_relevant"] or state["rewrites"] >= MAX_REWRITES:
            return "generate"
        return "rewrite"

    builder = StateGraph(State)
    for node in (classify, retrieve, grade, rewrite, generate, small_talk, draft_ticket, human_review, open_ticket):
        builder.add_node(node)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route_intent, ["retrieve", "draft_ticket", "small_talk"])
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges("grade", after_grade, ["generate", "rewrite"])
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("generate", END)
    builder.add_edge("small_talk", END)
    builder.add_edge("draft_ticket", "human_review")
    builder.add_edge("open_ticket", END)

    return builder.compile(checkpointer=checkpointer, name="docshelp")


def _context(state: State) -> str:
    return "\n\n".join(f"[source: {d['source']}]\n{d['content']}" for d in state["documents"])
