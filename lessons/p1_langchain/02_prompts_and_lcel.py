"""Prompt templates and LCEL: compose prompt -> model -> parser with `|`.

Academy: LangChain Essentials L1 "Create Agent" (the prompt/model building blocks it is made of).

Concepts:
  * ChatPromptTemplate: reusable prompts with {variables}
  * MessagesPlaceholder: slot a whole conversation into a prompt
  * few-shot prompting: example question/answer pairs shown to the model as messages
  * LCEL: `prompt | llm | parser` is itself a Runnable (invoke / stream / batch)
  * RunnableParallel and RunnableLambda: run branches side by side, and drop plain functions into a chain

Run:  uv run poe lesson 1.2
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableParallel

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header


def run(llm: BaseChatModel) -> dict:
    header("1. Prompt template | model | parser")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a support agent for {product}. Reply in at most two sentences."),
            ("human", "{question}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    print(chain.invoke({"product": "Nimbus Notes", "question": "Can I use the app offline?"}))
    print("Note: the model hasn't seen our docs, so it is guessing. Lesson 1.5 (RAG) fixes that.")

    header("2. MessagesPlaceholder + few-shot examples")
    examples = [
        {"q": "I was charged twice", "a": "Category: billing. I'm sorry! Refunds take 5-10 business days."},
        {"q": "My notes won't sync", "a": "Category: sync. Let's check your connection icon first."},
    ]
    few_shot = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=ChatPromptTemplate.from_messages([("human", "{q}"), ("ai", "{a}")]),
    )
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a Nimbus Notes support agent. Start with the category, like the examples."),
            few_shot,
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )
    history = [HumanMessage("Do you have a desktop app?"), AIMessage("Yes, for Windows and macOS.")]
    inputs = {"history": history, "question": "And Linux?"}
    print("What the model receives:")
    for m in chat_prompt.format_messages(**inputs):
        print(f"  {m.type:6} {m.text[:70]}")
    print(f"\n{(chat_prompt | llm | StrOutputParser()).invoke(inputs)}")

    header("3. RunnableParallel: two chains on the same input, at the same time")
    answer = ChatPromptTemplate.from_template("Answer briefly: {question}") | llm | StrOutputParser()
    title = (
        ChatPromptTemplate.from_template("Write a 3-word help-article title for: {question}")
        | llm
        | StrOutputParser()
        | RunnableLambda(lambda s: s.strip().strip('"'))  # any function can be a step
    )
    parallel = RunnableParallel(answer=answer, title=title).invoke({"question": "How does offline mode work?"})
    print(parallel)

    return {"parallel": parallel}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm())


if __name__ == "__main__":
    main()
