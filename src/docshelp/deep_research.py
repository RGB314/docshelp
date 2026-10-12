"""Deep research about Nimbus Notes: scope → research (in parallel) → report (Part 6.5–6.9).

  scope        clarify_with_user ──(needs clarification? ask and stop)──▶ write_research_brief
  research     supervisor ⇄ supervisor_tools ──▶ researcher subgraphs in parallel (asyncio.gather)
  researcher   llm_call ⇄ tools (search_sources, think) ──▶ compress_research
  report       final_report_generation

Each piece is a graph of its own, so the lessons can run and test them separately, and the full system
nests them as subgraphs. The research assistant of lesson 4.4 used the same map-reduce idea with a
fixed plan. Here a supervisor *decides* what to research next and when to stop.

Sources: the docs (retriever), the changelog, and optionally the web (`--web`, needs the `web` extra:
`uv sync --extra web`, free DuckDuckGo search via `ddgs`). Without it, research stays offline.
"""

from __future__ import annotations

import asyncio
import operator
from collections.abc import Sequence
from datetime import date
from typing import Annotated, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Checkpointer, Command
from pydantic import BaseModel, Field, ValidationError

from docshelp.config import structured_llm
from docshelp.ingest import format_docs
from docshelp.sources import search_changelog

# --- scoping ----------------------------------------------------------------------------------------


class ClarifyWithUser(BaseModel):
    need_clarification: bool = Field(description="True only if the request is too vague to research well.")
    question: str = Field(default="", description="One clarifying question, if needed.")
    verification: str = Field(
        default="", description="If no clarification is needed: confirm the scope in one sentence."
    )


class ResearchBrief(BaseModel):
    research_brief: str = Field(
        description="A detailed, self-contained brief: the question, scope, audience and output."
    )


CLARIFY_PROMPT = (
    "You scope research requests about the Nimbus Notes product for its product team. Today is {today}. "
    "Ask a clarifying question only if the request is genuinely ambiguous (who is it for, which aspect, what "
    "output). Don't ask if the conversation already answers it."
)
BRIEF_PROMPT = "Turn the conversation into a research brief for a team of researchers. Include every requirement."


class ScopeState(MessagesState):
    research_brief: str


def build_scoping_graph(llm: BaseChatModel, checkpointer: Checkpointer | None = None):
    def clarify_with_user(state: ScopeState) -> Command[Literal["write_research_brief", "__end__"]]:
        decision = structured_llm(llm, ClarifyWithUser).invoke(
            [SystemMessage(CLARIFY_PROMPT.format(today=date.today())), *state["messages"]]
        )
        if decision.need_clarification:  # ask, and end the run: the user's answer starts the next one
            return Command(goto=END, update={"messages": [AIMessage(decision.question)]})
        return Command(goto="write_research_brief", update={"messages": [AIMessage(decision.verification)]})

    def write_research_brief(state: ScopeState) -> dict:
        brief = structured_llm(llm, ResearchBrief).invoke([SystemMessage(BRIEF_PROMPT), *state["messages"]])
        return {"research_brief": brief.research_brief}

    builder = StateGraph(ScopeState)
    builder.add_node(clarify_with_user)
    builder.add_node(write_research_brief)
    builder.add_edge(START, "clarify_with_user")
    builder.add_edge("write_research_brief", END)
    return builder.compile(checkpointer=checkpointer, name="scoping")


# --- research tools ---------------------------------------------------------------------------------


@tool
def think(reflection: str) -> str:
    """Reflect before the next step: what did you learn, what's missing, should you search again or stop?"""
    return f"Reflection recorded: {reflection}"


def web_search(query: str, max_results: int = 3) -> str:
    try:
        from ddgs import DDGS
    except ImportError:
        return "(web search unavailable: install it with `uv sync --extra web`)"
    try:
        hits = DDGS().text(query, max_results=max_results)
    except Exception as err:  # noqa: BLE001 - network problems shouldn't stop the research
        return f"(web search failed: {err})"
    return "\n\n".join(f"[web: {h.get('href', '?')}] {h.get('title', '')}\n{h.get('body', '')}" for h in hits)


def make_research_tools(retriever: BaseRetriever, *, web: bool = False) -> list[BaseTool]:
    @tool
    def search_sources(query: str) -> str:
        """Search Nimbus Notes sources: the product docs, the release changelog (and the web, if enabled)."""
        parts = [format_docs(retriever.invoke(query))]
        parts += [f"[source: changelog.md] {entry}" for entry in search_changelog(query)]
        if web:
            parts.append(web_search(query))
        return "\n\n".join(p for p in parts if p)

    return [search_sources, think]


# --- the researcher ---------------------------------------------------------------------------------

RESEARCHER_PROMPT = (
    "You research one topic about Nimbus Notes. Use search_sources (several focused queries beat one broad "
    "one) and think after each search to decide whether you have enough. Stop when you can answer the topic "
    "well, or after a few searches. Your last message is not seen by anyone; just stop calling tools."
)
COMPRESS_PROMPT = (
    "Clean up the research above into a compact set of findings for the topic: {topic}. Keep every relevant "
    "fact and figure, with its source in brackets, e.g. [03-sync-and-offline.md]. Drop repetition and "
    "anything irrelevant. Don't add facts that aren't in the tool results."
)


class ResearcherState(MessagesState):
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], operator.add]


class ResearcherOutput(TypedDict):
    compressed_research: str
    raw_notes: list[str]


def _answered(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Drop a trailing AI message whose tool calls never ran, so any model API accepts the history."""
    if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return messages[:-1]
    return messages


def build_researcher(llm: BaseChatModel, tools: Sequence[BaseTool], *, max_tool_rounds: int = 4):
    model = llm.bind_tools(list(tools))

    def llm_call(state: ResearcherState) -> dict:
        return {"messages": [model.invoke([SystemMessage(RESEARCHER_PROMPT), *state["messages"]])]}

    def should_continue(state: ResearcherState) -> Literal["tools", "compress_research"]:
        rounds = sum(1 for m in state["messages"] if isinstance(m, AIMessage) and m.tool_calls)
        if state["messages"][-1].tool_calls and rounds <= max_tool_rounds:
            return "tools"
        return "compress_research"

    def compress_research(state: ResearcherState) -> dict:
        prompt = COMPRESS_PROMPT.format(topic=state["research_topic"])
        summary = llm.invoke([*_answered(state["messages"]), HumanMessage(prompt)]).text
        raw = [m.text for m in state["messages"] if isinstance(m, ToolMessage) and m.name != "think"]
        return {"compressed_research": summary, "raw_notes": ["\n".join(raw)]}

    builder = StateGraph(ResearcherState, output_schema=ResearcherOutput)
    builder.add_node(llm_call)
    builder.add_node("tools", ToolNode(list(tools)))  # sync or async tools (MCP), whichever the run uses
    builder.add_node(compress_research)
    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges("llm_call", should_continue)
    builder.add_edge("tools", "llm_call")
    builder.add_edge("compress_research", END)
    return builder.compile(name="researcher")


def research_input(topic: str) -> dict:
    return {"research_topic": topic, "messages": [HumanMessage(topic)]}


# --- the supervisor ---------------------------------------------------------------------------------


class ConductResearch(BaseModel):
    """Delegate one research topic to a researcher. Give a complete, standalone description of the topic."""

    research_topic: str


class ResearchComplete(BaseModel):
    """Call when the findings are enough to write the report."""


SUPERVISOR_PROMPT = (
    "You lead research for this brief:\n{brief}\n\n"
    "Delegate with ConductResearch: one call per distinct sub-topic, up to {max_concurrent} in parallel in one "
    "turn. Researchers can't see each other or this conversation, so describe each topic fully. Use think to "
    "plan and to review results. Call ResearchComplete as soon as the findings cover the brief."
)


class SupervisorState(TypedDict, total=False):
    supervisor_messages: Annotated[list[BaseMessage], add_messages]
    research_brief: str
    notes: Annotated[list[str], operator.add]
    raw_notes: Annotated[list[str], operator.add]
    research_iterations: int


def build_supervisor(llm: BaseChatModel, researcher, *, max_concurrent: int = 3, max_iterations: int = 3):
    model = llm.bind_tools([ConductResearch, ResearchComplete, think])

    def supervisor(state: SupervisorState) -> dict:
        response = model.invoke(state["supervisor_messages"])
        return {"supervisor_messages": [response], "research_iterations": state.get("research_iterations", 0) + 1}

    async def supervisor_tools(state: SupervisorState) -> Command[Literal["supervisor", "__end__"]]:
        last = state["supervisor_messages"][-1]
        calls = last.tool_calls
        done = (
            not calls
            or any(c["name"] == "ResearchComplete" for c in calls)
            or state.get("research_iterations", 0) >= max_iterations
        )
        if done:
            return Command(goto=END)
        replies: list[ToolMessage] = []
        research = []
        for call in calls:
            try:  # small models sometimes leave out required arguments: tell the model, don't crash the run
                if call["name"] == "think":
                    replies.append(ToolMessage(think.invoke(call["args"]), tool_call_id=call["id"], name="think"))
                elif call["name"] == "ConductResearch":
                    research.append((call, ConductResearch.model_validate(call["args"]).research_topic))
            except ValidationError as err:
                text = f"Error: invalid arguments for {call['name']}: {err.errors()[0]['msg']}. Try again."
                replies.append(ToolMessage(text, tool_call_id=call["id"], name=call["name"], status="error"))
        allowed, overflow = research[:max_concurrent], research[max_concurrent:]
        # The parallel step: each researcher is an independent subgraph run, awaited together.
        runs = [researcher.ainvoke(research_input(topic)) for _, topic in allowed]
        results = await asyncio.gather(*runs)
        for (call, _), result in zip(allowed, results):
            replies.append(ToolMessage(result["compressed_research"], tool_call_id=call["id"], name="ConductResearch"))
        for call, _ in overflow:
            text = f"Not run: at most {max_concurrent} researchers at a time. Ask again next turn if still needed."
            replies.append(ToolMessage(text, tool_call_id=call["id"], name="ConductResearch"))
        return Command(
            goto="supervisor",
            update={
                "supervisor_messages": replies,
                "notes": [r["compressed_research"] for r in results],
                "raw_notes": [n for r in results for n in r["raw_notes"]],
            },
        )

    builder = StateGraph(SupervisorState)
    builder.add_node(supervisor)
    builder.add_node(supervisor_tools)
    builder.add_edge(START, "supervisor")
    builder.add_edge("supervisor", "supervisor_tools")
    return builder.compile(name="research_supervisor")


def supervisor_input(brief: str, *, max_concurrent: int = 3) -> dict:
    system = SUPERVISOR_PROMPT.format(brief=brief, max_concurrent=max_concurrent)
    return {"research_brief": brief, "supervisor_messages": [SystemMessage(system), HumanMessage(brief)]}


# --- the full system --------------------------------------------------------------------------------

REPORT_PROMPT = (
    "Write a research report for this brief:\n{brief}\n\nFindings from the researchers:\n{findings}\n\n"
    "Use Markdown with a title, an executive summary, one section per theme, and a Sources section listing "
    "every [source] you cite. Use only the findings. Say plainly where the findings are thin. Today is {today}."
)


class DeepResearchState(MessagesState):
    research_brief: str
    supervisor_messages: Annotated[list[BaseMessage], add_messages]
    notes: Annotated[list[str], operator.add]
    raw_notes: Annotated[list[str], operator.add]
    final_report: str


def build_deep_research(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    checkpointer: Checkpointer | None = None,
    *,
    max_concurrent: int = 3,
    max_iterations: int = 3,
):
    scoping = build_scoping_graph(llm)
    researcher = build_researcher(llm, tools)
    supervisor = build_supervisor(llm, researcher, max_concurrent=max_concurrent, max_iterations=max_iterations)

    def scope(state: DeepResearchState) -> Command[Literal["research_supervisor", "__end__"]]:
        result = scoping.invoke({"messages": state["messages"]})
        new = result["messages"][len(state["messages"]):]
        if not result.get("research_brief"):  # a clarifying question: show it and wait for the answer
            return Command(goto=END, update={"messages": new})
        start = supervisor_input(result["research_brief"], max_concurrent=max_concurrent)
        return Command(goto="research_supervisor", update={"messages": new, **start})

    def final_report_generation(state: DeepResearchState) -> dict:
        findings = "\n\n---\n\n".join(state.get("notes", [])) or "(no findings)"
        report = llm.invoke(
            REPORT_PROMPT.format(brief=state["research_brief"], findings=findings, today=date.today())
        ).text
        return {"final_report": report, "messages": [AIMessage(report)]}

    builder = StateGraph(DeepResearchState)
    builder.add_node(scope)
    builder.add_node("research_supervisor", supervisor)  # a subgraph sharing brief, notes, messages
    builder.add_node(final_report_generation)
    builder.add_edge(START, "scope")
    builder.add_edge("research_supervisor", "final_report_generation")
    builder.add_edge("final_report_generation", END)
    return builder.compile(checkpointer=checkpointer, name="deep_research")
