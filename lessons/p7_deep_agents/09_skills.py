"""Skills: know-how the agent loads only when a task needs it.

Academy: Introduction to Deep Agents M3 "Skills".

Concepts:
  * a skill is a folder with a `SKILL.md`: YAML frontmatter (`name`, `description`) plus instructions, and
    optionally files next to it (scripts, templates). DocsHelp's are in `skills/` at the repo root:
      refund-policy/   when and how much to refund
      log-analysis/    how to read a sync log, plus `count_levels.py`
  * progressive disclosure: only each skill's name, description and path go into the system prompt. The
    model reads the full `SKILL.md` with `read_file` when a task matches the description. Dozens of skills
    cost a few lines of prompt each, not their whole text. So write the description for the model: when to use it.
  * `skills=["/skills/"]` lists sources: paths in the agent's backend. Here a `CompositeBackend` maps
    /skills/ to the folder on disk (read-only). Later sources override earlier ones with the same skill name,
    so you can layer team skills over shared ones.
  * skills vs tools vs memory: tools *do* things; skills are *how-to* text read on demand; memory (7.10) is
    always in the prompt.

Use case: a billing complaint, answered by the rules of the refund-policy skill.

Run:  uv run poe lesson 7.9
"""

from deepagents import FilesystemPermission
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.customers import Customer
from docshelp.deep import SKILLS_DIR, deep_agent
from docshelp.tools import my_account, quote_price

CUSTOMER = Customer(name="Ada", plan="team", account_id="acct-1001")
QUESTION = "We were charged for 12 seats but we only have 11 members. Can we get a refund?"


def run(llm: BaseChatModel) -> dict:
    backend = CompositeBackend(
        default=StateBackend(), routes={"/skills/": FilesystemBackend(root_dir=SKILLS_DIR, virtual_mode=True)}
    )
    agent = deep_agent(
        llm,
        tools=[my_account, quote_price],
        backend=backend,
        skills=["/skills/"],
        permissions=[FilesystemPermission(operations=["write"], paths=["/skills/**"], mode="deny")],
        system_prompt="You are Nimbus Notes billing support. Follow the matching skill exactly.",
        context_schema=Customer,
    )
    result = agent.invoke({"messages": [{"role": "user", "content": QUESTION}]}, context=CUSTOMER)
    skills_read = [
        call["args"]["file_path"]
        for message in result["messages"]
        for call in getattr(message, "tool_calls", [])
        if call["name"] == "read_file" and call["args"].get("file_path", "").endswith("SKILL.md")
    ]
    header(f"Skills the agent opened: {skills_read}")
    for message in result["messages"]:
        if isinstance(message, ToolMessage) and message.name != "read_file":
            print(f"  {message.name} -> {message.text[:100]}")
    header("Answer")
    print(result["messages"][-1].text)
    return {"skills_read": skills_read, "answer": result["messages"][-1].text, "messages": result["messages"]}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm())


if __name__ == "__main__":
    main()
