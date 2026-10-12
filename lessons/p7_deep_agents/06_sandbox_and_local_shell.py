"""Sandboxes and the local shell: letting an agent run commands, and how to do that safely.

Academy: Introduction to Deep Agents M2 "Sandboxes and LocalShell".

Concepts:
  * a backend that also implements `SandboxBackendProtocol` gives the agent an `execute` tool. With it the
    agent can write a script and run it: count, parse, test, instead of reasoning over raw text.
  * `LocalShellBackend(root_dir=...)` runs commands on *this machine* with `subprocess(shell=True)`.
    It is NOT a sandbox. `virtual_mode=True` confines the *file tools* to `root_dir`, but a shell command
    can still read, delete or upload anything your user account can. So, every time:
      1. a throwaway working directory, holding copies of only the files the task needs
      2. a minimal environment: `inherit_env=False` (the default) keeps your API keys out of reach;
         pass only what commands need, here `PATH` to this project's Python
      3. a `timeout` and a cap on output size
      4. a human or a policy approving each command (`interrupt_on={"execute": ...}`), as below
      5. in production, a real sandbox instead: a remote container or VM. Sandbox backends subclass
         `deepagents.backends.sandbox.BaseSandbox` (implement `execute`, `upload_files`, `download_files`);
         hosted sandbox providers ship such integrations as separate packages. The agent's tools don't
         change, and the worst case is a destroyed container, not your laptop.
  * here the policy is an allowlist reviewer: only `python <script>.py <file>` with plain file names is
    approved; anything else (`python -c ...`, pipes, network or delete commands) is rejected with a reason
    the model reads. An allowlist limits which *commands* run, not what the agent's script *does*: that's
    why points 1, 2 and 5 still matter.

Use case: count the errors in a customer's sync client log with a short Python script.

Run:  uv run poe lesson 7.6
"""

import os
import re
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from deepagents.backends import LocalShellBackend
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from docshelp.config import DATA_DIR, describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.deep import deep_agent

LOG = DATA_DIR / "logs" / "sync-client.log"
TASK = (
    "sync-client.log is a customer's sync log. Write a Python script /count.py that counts ERROR lines per "
    "message, run it with `python count.py sync-client.log`, then tell me the most common error."
)
ALLOWED = re.compile(r"python [\w-]+\.py( [\w.-]+)*")  # run a script from the work dir on plain file names


def allowlist(action: dict) -> dict:
    """Approve only commands matching the allowlist; reject the rest, saying why."""
    command = action["args"].get("command", "").strip()
    if ALLOWED.fullmatch(command):
        return {"type": "approve"}
    return {"type": "reject", "message": "Command not allowed: only `python <script>.py <file>` may run."}


def minimal_env() -> dict[str, str]:
    """Just enough for `python` to run: this project's Python first on PATH. No keys, no tokens."""
    env = {"PATH": str(Path(sys.executable).parent)}
    if "SYSTEMROOT" in os.environ:  # Windows needs it to start Python
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    return env


def run(llm: BaseChatModel, workdir: Path | None = None, review: Callable[[dict], dict] = allowlist) -> dict:
    workdir = Path(workdir or tempfile.mkdtemp(prefix="docshelp-shell-"))
    shutil.copy(LOG, workdir / LOG.name)  # only what the task needs
    backend = LocalShellBackend(
        root_dir=workdir, virtual_mode=True, timeout=30, max_output_bytes=20_000, env=minimal_env()
    )
    agent = deep_agent(
        llm,
        backend=backend,
        interrupt_on={"execute": {"allowed_decisions": ["approve", "reject"]}},
        checkpointer=InMemorySaver(),
        system_prompt="You analyse logs by writing and running small Python scripts. Be brief.",
    )
    config = {"configurable": {"thread_id": "shell"}}
    header(f"Working directory: {workdir.name} (a temp dir)")
    result = agent.invoke({"messages": [{"role": "user", "content": TASK}]}, config)
    commands = []
    while "__interrupt__" in result:
        decisions = []
        for action in result["__interrupt__"][0].value["action_requests"]:
            decisions.append(review(action))
            commands.append((action["args"]["command"], decisions[-1]["type"]))
            print(f"  $ {action['args']['command']}  -> {decisions[-1]['type']}")
        result = agent.invoke(Command(resume={"decisions": decisions}), config)
    outputs = [m.text for m in result["messages"] if isinstance(m, ToolMessage) and m.name == "execute"]
    for output in outputs:
        print(output[:400])
    header("Answer")
    print(result["messages"][-1].text)
    return {"workdir": workdir, "commands": commands, "outputs": outputs, "answer": result["messages"][-1].text}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm())


if __name__ == "__main__":
    main()
