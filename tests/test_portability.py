"""Docker / Dev Container / line-ending setup that makes the project behave the same on every OS."""

import json
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_line_endings_are_normalised_to_lf():
    lines = read(".gitattributes").splitlines()
    assert "* text=auto eol=lf" in lines


def test_dockerfile_uses_locked_dependencies_and_matching_python():
    dockerfile = read("Dockerfile")
    python = read(".python-version").strip()
    assert f"python{python}" in dockerfile
    assert "uv sync --frozen" in dockerfile
    assert "PYTHONUTF8=1" in dockerfile
    assert "COPY uv.lock" in dockerfile or "uv.lock" in dockerfile


def test_dockerignore_keeps_secrets_and_local_state_out_of_the_image():
    ignored = read(".dockerignore").splitlines()
    for entry in (".env", ".venv", ".cache", ".git", ".claude"):
        assert entry in ignored


def compose() -> dict:
    return yaml.safe_load(read("compose.yaml"))


def test_compose_defines_app_server_and_optional_ollama():
    services = compose()["services"]
    assert {"app", "server", "server-auth", "ollama"} <= set(services)
    assert services["ollama"]["profiles"] == ["ollama"]
    assert "2024:2024" in services["server"]["ports"]
    assert "2025:2025" in services["server-auth"]["ports"]
    assert "langgraph.auth.json" in services["server-auth"]["command"]


def test_compose_reads_env_file_without_requiring_it():
    services = compose()["services"]
    for name in ("app", "server"):
        env_files = services[name]["env_file"]
        assert {"path": ".env", "required": False} in env_files


def test_app_container_talks_to_the_server_and_ollama_by_service_name():
    env = compose()["services"]["app"]["environment"]
    assert env["LANGGRAPH_URL"] == "http://server:2024"
    assert env["LANGGRAPH_AUTH_URL"] == "http://server-auth:2025"
    assert "ollama:11434" in env["OLLAMA_BASE_URL"]


def test_server_has_a_healthcheck_so_compose_can_wait_for_it():
    for service, port in (("server", 2024), ("server-auth", 2025)):
        healthcheck = compose()["services"][service]["healthcheck"]
        assert f"{port}/ok" in " ".join(healthcheck["test"])


def workflow() -> dict:
    return yaml.safe_load(read(".github/workflows/ci.yml"))


def test_ci_runs_the_suites_natively_on_all_three_operating_systems():
    job = workflow()["jobs"]["test"]
    assert set(job["strategy"]["matrix"]["os"]) == {"ubuntu-latest", "windows-latest", "macos-latest"}
    steps = " ".join(str(step.get("run", "")) for step in job["steps"])
    assert "uv sync --frozen --all-extras" in steps
    assert "poe test" in steps and "poe e2e" in steps


def test_ci_runs_docker_e2e_with_a_real_local_model():
    job = workflow()["jobs"]["docker-e2e-shard"]
    steps = " ".join(str(step.get("run", "")) for step in job["steps"])
    assert "docker compose build" in steps
    assert "--profile ollama" in steps
    assert "DOCSHELP_E2E_MODEL" in steps
    assert "poe e2e" in steps
    assert "up -d --wait server server-auth" in steps
    assert "poe lesson 8.3" in steps and "poe lesson 8.10" in steps  # Part 8 clients over the compose network


def test_docker_e2e_shards_cover_every_part_once_behind_one_required_check():
    from docshelp.tasks import find_lessons

    jobs = workflow()["jobs"]
    shards = jobs["docker-e2e-shard"]["strategy"]["matrix"]["include"]
    parts = [part for shard in shards for part in shard["parts"].split(",")]
    assert sorted(parts) == sorted({code.split(".")[0] for code, _ in find_lessons()})  # each part exactly once
    assert sum(bool(shard.get("full")) for shard in shards) == 1  # unit, keyless and compose checks run once
    gate = jobs["docker-e2e"]  # the name branch protection requires
    assert gate["needs"] == "docker-e2e-shard" and gate["if"] == "always()"


def test_ci_triggers_on_pull_requests_to_main():
    triggers = workflow()[True]  # YAML 1.1 parses the key `on` as True
    assert "main" in triggers["pull_request"]["branches"]


def test_devcontainer_reuses_the_project_setup():
    config = json.loads(read(".devcontainer/devcontainer.json"))
    assert "uv sync" in config["postCreateCommand"]
    assert 2024 in config["forwardPorts"]


def test_container_images_install_every_provider_extra():
    extras = tomllib.loads(read("pyproject.toml"))["project"]["optional-dependencies"]
    assert extras
    assert "--all-extras" in read("Dockerfile")
    assert "--all-extras" in json.loads(read(".devcontainer/devcontainer.json"))["postCreateCommand"]
