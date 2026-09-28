"""Cross-shell developer tasks: the same `uv run poe <task>` works in PowerShell, cmd, Git Bash, macOS and Linux."""

import tomllib
from pathlib import Path

import pytest

from docshelp import tasks

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_init_env_copies_the_example(tmp_path):
    (tmp_path / ".env.example").write_text("DOCSHELP_PROVIDER=anthropic\n", encoding="utf-8")
    assert tasks.init_env(tmp_path) is True
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "DOCSHELP_PROVIDER=anthropic\n"


def test_init_env_never_overwrites_an_existing_env(tmp_path):
    (tmp_path / ".env.example").write_text("A=\n", encoding="utf-8")
    (tmp_path / ".env").write_text("GROQ_API_KEY=my-secret\n", encoding="utf-8")
    assert tasks.init_env(tmp_path) is False
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "GROQ_API_KEY=my-secret\n"


def test_doctor_reports_every_provider_and_marks_the_selected_one():
    rows = {r.name: r for r in tasks.doctor({"DOCSHELP_PROVIDER": "groq", "GROQ_API_KEY": "k"})}

    assert set(rows) == {"anthropic", "openai", "google", "groq", "ollama"}
    assert rows["groq"].selected and rows["groq"].ready
    assert not rows["anthropic"].selected
    assert not rows["anthropic"].ready and "ANTHROPIC_API_KEY" in rows["anthropic"].message


def test_doctor_runs_the_runtime_check_for_the_selected_provider_only():
    env = {"DOCSHELP_PROVIDER": "ollama", "OLLAMA_BASE_URL": "http://127.0.0.1:9"}
    rows = {r.name: r for r in tasks.doctor(env)}
    assert not rows["ollama"].ready
    assert "Can't reach Ollama" in rows["ollama"].message


def test_doctor_main_exit_code_reflects_the_selected_provider(capsys):
    assert tasks.doctor_main({"DOCSHELP_PROVIDER": "groq", "GROQ_API_KEY": "k"}) == 0
    assert tasks.doctor_main({"DOCSHELP_PROVIDER": "openai"}) == 1
    out = capsys.readouterr().out
    assert "groq" in out and "OPENAI_API_KEY" in out


def test_poethepoet_is_a_dev_dependency():
    assert any(dep.startswith("poethepoet") for dep in PYPROJECT["dependency-groups"]["dev"])


def test_poe_tasks_cover_the_whole_workflow():
    poe_tasks = PYPROJECT["tool"]["poe"]["tasks"]
    for name in ("init-env", "doctor", "test", "server", "stage1", "stage2", "stage3", "stage4", "stage5"):
        assert name in poe_tasks, f"missing poe task {name!r}"


def test_unit_and_e2e_suites_are_separate_tasks():
    poe_tasks = PYPROJECT["tool"]["poe"]["tasks"]
    assert "not e2e" in poe_tasks["test"]["cmd"]
    assert "-m e2e" in poe_tasks["e2e"]["cmd"]
    markers = PYPROJECT["tool"]["pytest"]["ini_options"]["markers"]
    assert any(m.startswith("e2e:") for m in markers)


@pytest.mark.parametrize("n", range(1, 6))
def test_stage_tasks_point_at_existing_scripts(n):
    task = PYPROJECT["tool"]["poe"]["tasks"][f"stage{n}"]
    cmd = task["cmd"] if isinstance(task, dict) else task
    script = cmd.split()[-1]
    assert (ROOT / script).is_file()
    assert "\\" not in cmd  # forward slashes work in every shell
