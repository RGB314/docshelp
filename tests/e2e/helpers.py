"""Helpers for end-to-end tests: run lessons and stages as real processes, start a real LangGraph Server."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Iterator, Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROVIDER_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
}


def base_env(**overrides: str) -> dict[str, str]:
    """A child-process environment where no real key can leak in, even from a developer's .env.

    Every key var is set (to empty unless overridden): python-dotenv never overrides variables that
    already exist, so values in .env are ignored.
    """
    env = dict(os.environ)
    for var in (*PROVIDER_KEYS.values(), "GEMINI_API_KEY", "LANGSMITH_API_KEY"):
        env[var] = ""
    env.update({"LANGSMITH_TRACING": "false", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    env.update(overrides)
    return env


def run_lesson(code: str, env: Mapping[str, str], *args: str, timeout: int = 180) -> subprocess.CompletedProcess:
    """Run a lesson exactly as a learner does: through the `poe lesson` runner, in a fresh process."""
    runner = "import sys; from docshelp.tasks import lesson_main; lesson_main(sys.argv[1:])"
    return _run([sys.executable, "-c", runner, code, *args], env, timeout)


def run_evals(env: Mapping[str, str], timeout: int = 600) -> subprocess.CompletedProcess:
    """`uv run poe evals`: the eval suite in `evals/`, against the configured model."""
    return _run([sys.executable, "-m", "pytest", "evals", "-p", "no:cacheprovider", "-q"], env, timeout)


def _run(command: list[str], env: Mapping[str, str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=ROOT,
        env=dict(env),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        stdin=subprocess.DEVNULL,
    )


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _langgraph_executable() -> str:
    scripts = Path(sys.executable).parent
    for name in ("langgraph", "langgraph.exe"):
        if (scripts / name).exists():
            return str(scripts / name)
    return "langgraph"


@contextlib.contextmanager
def langgraph_server(
    config_dir: Path, env: Mapping[str, str], startup_timeout: int = 120, config: str = "langgraph.json"
) -> Iterator[str]:
    """Run `langgraph dev` for the `config` file in `config_dir`; yield its base URL."""
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    log = Path(tempfile.mkdtemp()) / "server.log"  # never inside the repo
    with log.open("w", encoding="utf-8") as out:
        proc = subprocess.Popen(
            [_langgraph_executable(), "dev", "--config", config, "--port", str(port), "--no-browser", "--no-reload"],
            cwd=config_dir,
            env=dict(env),
            stdout=out,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + startup_timeout
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"langgraph dev exited early:\n{log.read_text(encoding='utf-8')}")
                try:
                    urllib.request.urlopen(f"{url}/ok", timeout=2)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"langgraph dev did not start:\n{log.read_text(encoding='utf-8')}")
                    time.sleep(1)
            yield url
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()


FAKE_APP = Path(__file__).parent / "fake_server" / "fake_app.py"


def fake_server_dir(target: Path, config: str = "langgraph.json") -> Path:
    """A server folder for `langgraph dev` that serves the repo's `config` with the fakes in fake_app.py.

    The graphs, routes, auth and store index keep their names; every path points at fake_app.py, which
    re-exports the real objects built with a fake model. So the fake server can't drift from the real one.
    """
    real = json.loads((ROOT / config).read_text(encoding="utf-8"))

    def fake(spec: str) -> str:
        return f"./fake_app.py:{spec.rsplit(':', 1)[1]}"

    fake_config: dict = {
        "dependencies": ["."],
        "python_version": real["python_version"],
        "graphs": {name: fake(spec) for name, spec in real["graphs"].items()},
    }
    if "http" in real:
        fake_config["http"] = {**real["http"], "app": fake(real["http"]["app"])}
    if "store" in real:
        fake_config["store"] = {"index": {**real["store"]["index"], "embed": fake(real["store"]["index"]["embed"])}}
    if "auth" in real:
        fake_config["auth"] = {**real["auth"], "path": fake(real["auth"]["path"])}
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy(FAKE_APP, target / "fake_app.py")
    (target / "langgraph.json").write_text(json.dumps(fake_config, indent=2), encoding="utf-8")
    return target


def ollama_ready(base_url: str, model: str) -> bool:
    """True if an Ollama server is reachable at `base_url` and has `model` pulled."""
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=3) as response:
            names = [m["name"] for m in json.load(response).get("models", [])]
    except OSError:
        return False
    return model in names or f"{model}:latest" in names
