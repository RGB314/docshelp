"""Guards that keep the repo safe and portable to share publicly."""

import re
import tomllib
from pathlib import Path

import pytest

from docshelp.providers import registry

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".venv", ".cache", ".git", ".pytest_cache", ".langgraph_api", "__pycache__", ".claude"}
TEXT_SUFFIXES = {
    ".py", ".md", ".toml", ".json", ".txt", ".example", ".yml", ".yaml", ".cfg", ".lock", ".sh", ".ps1",
    ".gitignore", ".gitattributes", ".dockerignore", "Dockerfile",
}

# Windows drive paths (X:\... or X:/...) and Unix home directories. `/home/vscode` is the standard
# user inside the Dev Container image (identical for everyone), so it isn't machine-specific.
LOCAL_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]+\w|/Users/\w|/home/(?!vscode\b)\w")


def repo_text_files():
    for path in ROOT.rglob("*"):
        if path.is_file() and not SKIP_DIRS & set(path.relative_to(ROOT).parts):
            if path.suffix in TEXT_SUFFIXES or path.name in TEXT_SUFFIXES:
                yield path


def test_no_local_machine_paths_in_repo():
    offenders = [
        f"{path.relative_to(ROOT)}:{n}: {line.strip()}"
        for path in repo_text_files()
        for n, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1)
        if LOCAL_PATH.search(line) and path.name != "test_repo_hygiene.py"
    ]
    assert offenders == []


def test_secrets_and_local_files_are_git_ignored():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for entry in (".env", ".venv/", ".cache/", ".claude/settings.local.json", "data/tickets.jsonl"):
        assert entry in ignored


def env_example() -> dict[str, str]:
    values = {}
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        line = line.lstrip("# ").strip()
        if re.match(r"^[A-Z_]+=", line):
            key, _, value = line.partition("=")
            values[key] = value
    return values


def test_env_example_documents_every_provider_and_setting():
    documented = env_example()
    for name in registry.names():
        provider = registry.create(name, env={})
        for var in provider.api_key_envs[:1]:
            assert var in documented, f"{var} missing from .env.example"
    for var in ("DOCSHELP_PROVIDER", "DOCSHELP_MODEL", "DOCSHELP_FAST_MODEL", "OLLAMA_BASE_URL", "LANGSMITH_API_KEY"):
        assert var in documented


def test_env_example_contains_no_secret_values():
    for key, value in env_example().items():
        if key.endswith("_API_KEY"):
            assert value == "", f"{key} has a value in .env.example"


def test_every_provider_has_a_matching_pyproject_extra():
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    for name in registry.names():
        assert registry.create(name, env={}).extra in extras


def test_agent_instructions_exist_and_claude_md_reuses_agents_md():
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "@AGENTS.md" in claude
    for must_mention in ("uv run poe test", "uv run poe e2e", "DOCSHELP_PROVIDER", "pull request"):
        assert must_mention in agents


def test_license_is_declared():
    assert (ROOT / "LICENSE").is_file()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["license"] == "Apache-2.0"


@pytest.mark.parametrize(
    "path", sorted((ROOT / "stages").glob("*.py")) + [ROOT / "src" / "docshelp" / "graph.py"], ids=lambda p: p.name
)
def test_app_code_is_provider_neutral(path):
    source = path.read_text(encoding="utf-8")
    for forbidden in ("require_anthropic_key", "anthropic:", 'method="json_schema"', "claude-"):
        assert forbidden not in source, f"{path.name} hard-codes {forbidden!r}"
