"""CURRICULUM.md maps every LangChain Academy lesson to a DocsHelp lesson; keep it truthful."""

import re
from pathlib import Path

from docshelp.tasks import find_lessons

ROOT = Path(__file__).resolve().parents[1]
CURRICULUM = (ROOT / "CURRICULUM.md").read_text(encoding="utf-8")
LESSON_REF = re.compile(r"`(lessons/p\d+_[\w]+/\d\d_[\w]+\.py)`")
PLANNED = re.compile(r"Part (\d+) \(planned\)")


def lesson_files() -> list[str]:
    return [path.relative_to(ROOT).as_posix() for _, path in find_lessons()]


def test_every_lesson_is_listed_in_the_curriculum():
    missing = [f for f in lesson_files() if f"`{f}`" not in CURRICULUM]
    assert missing == []


def test_every_referenced_lesson_exists():
    broken = [ref for ref in LESSON_REF.findall(CURRICULUM) if not (ROOT / ref).is_file()]
    assert broken == []


def test_parts_that_have_shipped_have_no_planned_rows_left():
    shipped = {int(f.split("/")[1][1:].split("_")[0]) for f in lesson_files()}
    still_planned = {int(n) for n in PLANNED.findall(CURRICULUM)}
    assert shipped & still_planned == set(), "a shipped part still has '(planned)' rows in CURRICULUM.md"


def test_every_part_has_self_check_questions_once_it_ships():
    shipped = sorted({int(f.split("/")[1][1:].split("_")[0]) for f in lesson_files()})
    for part in shipped:
        assert re.search(rf"^### Part {part} self-check", CURRICULUM, flags=re.M), f"Part {part} needs a self-check"
