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
    for name in ("init-env", "doctor", "test", "server", "lessons", "lesson", "evals", "stage5"):
        assert name in poe_tasks, f"missing poe task {name!r}"


def test_lessons_are_found_by_part_and_number():
    lessons = tasks.find_lessons()
    assert lessons, "no lessons found"
    codes = [code for code, _ in lessons]
    assert codes == sorted(codes, key=lambda c: tuple(map(int, c.split("."))))
    assert "1.1" in codes
    assert tasks.lesson_path("1.1").name.startswith("01_")
    assert tasks.lesson_path("01.01") == tasks.lesson_path("1.1")  # leading zeros are fine


def test_every_lesson_has_a_title_and_a_main():
    for code, path in tasks.find_lessons():
        module = tasks.load_lesson(code)
        assert tasks.lesson_title(path), f"{path.name} needs a docstring"
        assert callable(getattr(module, "main", None)), f"{path.name} needs main()"
        assert callable(getattr(module, "run", None)), f"{path.name} needs run() for offline tests"


def test_unknown_lesson_lists_the_options(capsys):
    with pytest.raises(SystemExit) as exit_info:
        tasks.lesson_main(["9.99"])
    assert exit_info.value.code == 1
    assert "1.1" in capsys.readouterr().err


def test_lessons_main_prints_every_lesson(capsys):
    tasks.lessons_main()
    out = capsys.readouterr().out
    for code, _ in tasks.find_lessons():
        assert f"{code} " in out


def test_unit_and_e2e_suites_are_separate_tasks():
    poe_tasks = PYPROJECT["tool"]["poe"]["tasks"]
    assert "not e2e" in poe_tasks["test"]["cmd"]
    assert "-m e2e" in poe_tasks["e2e"]["cmd"]
    markers = PYPROJECT["tool"]["pytest"]["ini_options"]["markers"]
    assert any(m.startswith("e2e:") for m in markers)


@pytest.mark.parametrize("n", range(5, 6))  # stages 1-4 became Parts 1, 3 and 5; stage 5 follows in Part 8
def test_stage_tasks_point_at_existing_scripts(n):
    task = PYPROJECT["tool"]["poe"]["tasks"][f"stage{n}"]
    cmd = task["cmd"] if isinstance(task, dict) else task
    script = cmd.split()[-1]
    assert (ROOT / script).is_file()
    assert "\\" not in cmd  # forward slashes work in every shell
