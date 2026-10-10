"""External memory: a SQLite checkpointer, so conversations survive restarts.

Academy: Intro to LangGraph M2 L6 "Chatbot w/ Summarizing Messages and External Memory".

Concepts:
  * `InMemorySaver` forgets everything when the process exits. A durable checkpointer stores each
    thread's checkpoints in a database: `SqliteSaver` (package `langgraph-checkpoint-sqlite`) locally,
    `PostgresSaver` in production. The graph code doesn't change, only the checkpointer passed in.
  * here the summarizing chatbot from lesson 3.9 talks in "session 1", the connection is closed (as if
    the app restarted), and "session 2" opens the same file and continues the same `thread_id`
  * the database also keeps every checkpoint, which is what time travel (lesson 3.14) reads
  * LangGraph Server (Part 8) brings its own durable persistence, so you don't pass a checkpointer there

Run:  uv run poe lesson 3.10      (the database is data/checkpoints.sqlite)
"""

import uuid
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.sqlite import SqliteSaver

from docshelp.chatbot import build_summarizing_chatbot
from docshelp.config import DATA_DIR, describe, ensure_llm_ready, get_llm
from docshelp.console import header


def say(bot, config: dict, text: str) -> None:
    reply = bot.invoke({"messages": [{"role": "user", "content": text}]}, config)["messages"][-1].text
    print(f"User: {text}\nDocsHelp: {reply}")


def run(llm: BaseChatModel, db_path: Path | None = None) -> dict:
    db_path = db_path or DATA_DIR / "checkpoints.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    header(f"Session 1 (writes to {db_path.name})")
    with SqliteSaver.from_conn_string(str(db_path)) as saver:
        bot = build_summarizing_chatbot(llm, saver)
        say(bot, config, "Hi, I'm Ada. I use Nimbus Notes on Windows.")
        say(bot, config, "Please remember that, I'll be back after lunch.")
    print("(connection closed: the process could exit here)")

    header("Session 2 (a fresh connection to the same file)")
    with SqliteSaver.from_conn_string(str(db_path)) as saver:
        bot = build_summarizing_chatbot(llm, saver)
        restored = len(bot.get_state(config).values["messages"])
        checkpoints = len(list(saver.list(config)))
        print(f"Loaded {restored} messages and {checkpoints} checkpoints for this thread from disk.")
        say(bot, config, "Hi again! Which device did I say I use?")
    return {"restored_messages": restored, "checkpoints": checkpoints}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm())


if __name__ == "__main__":
    main()
