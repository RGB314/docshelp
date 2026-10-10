"""A research assistant that writes a help-center article from several perspectives (Part 4.4).

        START → create_analysts → human_feedback ──(feedback?)──▶ create_analysts
                                       │ approved
                                       ▼  Send, one per analyst (map)
                                   interview  (a subgraph: ask → search docs ∥ changelog → answer ↺ → write_section)
                                       ▼  fan-out
              write_introduction ∥ write_body ∥ write_conclusion
                                       ▼  fan-in
                                    finalize → END

It combines Part 4's patterns: a human-in-the-loop step that can loop back (lesson 3.13), map-reduce
with `Send` (4.3), a subgraph (4.2) with parallel searches inside it (4.1), and parallel writers joined
by a final step. Part 6's deep-research agent scales the same idea up.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Checkpointer, Send, interrupt
from pydantic import BaseModel, Field

from docshelp.config import structured_llm
from docshelp.sources import search_changelog


class Analyst(BaseModel):
    name: str = Field(description="A first name.")
    role: str = Field(description="Who they represent, e.g. 'workspace admin'.")
    focus: str = Field(description="What this reader most needs from the article.")

    @property
    def persona(self) -> str:
        return f"{self.name}, a {self.role}. Focus: {self.focus}"


class Perspectives(BaseModel):
    analysts: list[Analyst] = Field(description="Distinct reader perspectives for the article.")


class ResearchState(TypedDict, total=False):
    topic: str
    max_analysts: int
    feedback: str
    analysts: list[Analyst]
    sections: Annotated[list[str], operator.add]  # one per interview
    sources: Annotated[list[str], operator.add]
    introduction: str
    body: str
    conclusion: str
    article: str


class InterviewState(MessagesState):
    topic: str
    analyst: Analyst
    max_turns: int
    context: Annotated[list[str], operator.add]  # the two searches write here in parallel
    sections: list[str]
    sources: list[str]


class InterviewOutput(TypedDict):  # all the parent gets back from an interview
    sections: list[str]
    sources: list[str]


def build_interview_graph(llm: BaseChatModel, retriever: BaseRetriever):
    def ask_question(state: InterviewState) -> dict:
        system = (
            f"You are {state['analyst'].persona}. You are interviewing a Nimbus Notes product expert for a "
            f"help-center article about: {state['topic']}. Ask one short, specific question you still need answered."
        )
        question = llm.invoke([SystemMessage(system), *state["messages"]]).text
        return {"messages": [HumanMessage(question, name="analyst")]}

    def latest_question(state: InterviewState) -> str:
        return next(m.text for m in reversed(state["messages"]) if isinstance(m, HumanMessage))

    def search_docs(state: InterviewState) -> dict:
        docs = retriever.invoke(latest_question(state))
        return {
            "context": [f"[{d.metadata.get('source', '?')}] {d.page_content}" for d in docs],
            "sources": sorted({d.metadata.get("source", "?") for d in docs}),
        }

    def search_changelog_node(state: InterviewState) -> dict:
        return {"context": [f"[changelog.md] {entry}" for entry in search_changelog(latest_question(state))]}

    def answer_question(state: InterviewState) -> dict:
        system = (
            "You are a Nimbus Notes product expert. Answer the interviewer's question using only this context, "
            f"citing sources in brackets.\n\n" + "\n\n".join(state["context"])
        )
        answer = llm.invoke([SystemMessage(system), *state["messages"]]).text
        return {"messages": [AIMessage(answer, name="expert")]}

    def next_step(state: InterviewState) -> Literal["ask_question", "write_section"]:
        answers = sum(1 for m in state["messages"] if isinstance(m, AIMessage) and m.name == "expert")
        return "write_section" if answers >= state["max_turns"] else "ask_question"

    def write_section(state: InterviewState) -> dict:
        transcript = "\n".join(f"{m.name}: {m.text}" for m in state["messages"])
        prompt = (
            f"Write a short help-center section (under 150 words) for {state['analyst'].role}s about "
            f"{state['topic']}, based on this interview. Keep the [source] citations.\n\n{transcript}"
        )
        heading = f"### For {state['analyst'].role}s"
        return {"sections": [f"{heading}\n\n{llm.invoke(prompt).text.strip()}"], "sources": state.get("sources", [])}

    builder = StateGraph(InterviewState, output_schema=InterviewOutput)
    builder.add_node(ask_question)
    builder.add_node("search_docs", search_docs)
    builder.add_node("search_changelog", search_changelog_node)
    builder.add_node(answer_question)
    builder.add_node(write_section)
    builder.add_edge(START, "ask_question")
    builder.add_edge("ask_question", "search_docs")  # two searches in parallel...
    builder.add_edge("ask_question", "search_changelog")
    builder.add_edge(["search_docs", "search_changelog"], "answer_question")  # ...joined before answering
    builder.add_conditional_edges("answer_question", next_step)
    builder.add_edge("write_section", END)
    return builder.compile()


def build_research_graph(
    llm: BaseChatModel, retriever: BaseRetriever, checkpointer: Checkpointer | None = None, *, max_turns: int = 1
):
    planner = structured_llm(llm, Perspectives)

    def create_analysts(state: ResearchState) -> dict:
        prompt = (
            f"Create {state.get('max_analysts', 2)} reader perspectives for a Nimbus Notes help-center article "
            f"about: {state['topic']}. Make them clearly different (e.g. end user, admin, security)."
        )
        if state.get("feedback"):
            prompt += f"\nApply this editor feedback: {state['feedback']}"
        analysts = planner.invoke(prompt).analysts[: state.get("max_analysts", 2)]
        # Small models sometimes return an empty list; one generic reader keeps the pipeline going.
        return {"analysts": analysts or [Analyst(name="Sam", role="Nimbus Notes user", focus=state["topic"])]}

    def human_feedback(state: ResearchState) -> dict:
        # Pause so an editor can approve the perspectives (empty answer) or ask for changes.
        answer = interrupt({"question": "Feedback on these perspectives? (empty to approve)",
                            "analysts": [a.persona for a in state["analysts"]]})
        return {"feedback": str(answer or "").strip()}

    def start_interviews(state: ResearchState):
        if state.get("feedback"):
            return "create_analysts"  # loop back with the feedback
        opening = HumanMessage(f"Let's talk about {state['topic']}.", name="analyst")
        return [
            Send("interview", {"topic": state["topic"], "analyst": a, "max_turns": max_turns, "messages": [opening]})
            for a in state["analysts"]
        ]

    def sections(state: ResearchState) -> str:
        return "\n\n".join(state["sections"])

    def write_introduction(state: ResearchState) -> dict:
        prompt = (
            f"Write a 2-sentence introduction for a help article about {state['topic']} covering:\n{sections(state)}"
        )
        return {"introduction": llm.invoke(prompt).text.strip()}

    def write_body(state: ResearchState) -> dict:
        prompt = (
            f"Merge these sections into one well-organised article body about {state['topic']}. Keep the "
            f"### headings and [source] citations; remove repetition.\n\n{sections(state)}"
        )
        return {"body": llm.invoke(prompt).text.strip()}

    def write_conclusion(state: ResearchState) -> dict:
        prompt = f"Write a 2-sentence conclusion with next steps for a help article covering:\n{sections(state)}"
        return {"conclusion": llm.invoke(prompt).text.strip()}

    def finalize(state: ResearchState) -> dict:
        sources = "\n".join(f"- {s}" for s in sorted(set(state.get("sources", [])) | {"changelog.md"}))
        article = (
            f"# {state['topic']}\n\n{state['introduction']}\n\n{state['body']}\n\n"
            f"## Next steps\n\n{state['conclusion']}\n\n## Sources\n\n{sources}"
        )
        return {"article": article}

    builder = StateGraph(ResearchState)
    builder.add_node(create_analysts)
    builder.add_node(human_feedback)
    builder.add_node("interview", build_interview_graph(llm, retriever))
    for writer in (write_introduction, write_body, write_conclusion):
        builder.add_node(writer)
        builder.add_edge("interview", writer.__name__)
    builder.add_node(finalize)
    builder.add_edge(START, "create_analysts")
    builder.add_edge("create_analysts", "human_feedback")
    builder.add_conditional_edges("human_feedback", start_interviews, ["create_analysts", "interview"])
    builder.add_edge(["write_introduction", "write_body", "write_conclusion"], "finalize")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer, name="research_assistant")
