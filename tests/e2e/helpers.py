"""Helpers for end-to-end tests: run stages as real processes, start a real LangGraph Server."""

from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator, Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGES = ROOT / "stages"
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


def run_stage(script: str, env: Mapping[str, str], *args: str, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(STAGES / script), *args],
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
def langgraph_server(config_dir: Path, env: Mapping[str, str], startup_timeout: int = 120) -> Iterator[str]:
    """Run `langgraph dev` for the langgraph.json in `config_dir`; yield its base URL."""
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    log = config_dir / "server.log"
    with log.open("w", encoding="utf-8") as out:
        proc = subprocess.Popen(
            [_langgraph_executable(), "dev", "--port", str(port), "--no-browser", "--no-reload"],
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


def ollama_ready(base_url: str, model: str) -> bool:
    """True if an Ollama server is reachable at `base_url` and has `model` pulled."""
    import json

    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=3) as response:
            names = [m["name"] for m in json.load(response).get("models", [])]
    except OSError:
        return False
    return model in names or f"{model}:latest" in names
