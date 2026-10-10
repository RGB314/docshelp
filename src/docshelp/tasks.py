"""Small, shell-independent helpers behind the `uv run poe <task>` commands.

Written in Python rather than PowerShell or bash so they behave identically on Windows, macOS and Linux.
"""

from __future__ import annotations

import ast
import importlib.util
import runpy
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from docshelp.config import PROJECT_ROOT, Settings
from docshelp.console import ensure_utf8
from docshelp.providers import ProviderConfigError, registry


def init_env(root: Path = PROJECT_ROOT) -> bool:
    """Create `.env` from `.env.example`. Never overwrites an existing `.env`. Returns True if created."""
    target = root / ".env"
    if target.exists():
        return False
    shutil.copyfile(root / ".env.example", target)
    return True


def init_env_main() -> None:
    ensure_utf8()
    if init_env():
        print("Created .env from .env.example. Set DOCSHELP_PROVIDER and that provider's API key in it.")
    else:
        print(".env already exists; left it unchanged.")


@dataclass(frozen=True)
class ProviderStatus:
    name: str
    display_name: str
    selected: bool
    ready: bool
    message: str


def doctor(env: Mapping[str, str] | None = None) -> list[ProviderStatus]:
    """Check every provider's setup. Only the selected one gets a runtime check (e.g. is Ollama running?)."""
    settings = Settings.from_env(env)
    rows = []
    for name in registry.names():
        provider = registry.create(name, env)
        selected = name == settings.provider
        try:
            provider.check()
            if selected:
                provider.check_runtime(settings.model)
            rows.append(ProviderStatus(name, provider.display_name, selected, True, "ready"))
        except ProviderConfigError as err:
            rows.append(ProviderStatus(name, provider.display_name, selected, False, str(err)))
    return rows


def doctor_main(env: Mapping[str, str] | None = None) -> int:
    ensure_utf8()
    rows = doctor(env)
    print("Provider setup (select one with DOCSHELP_PROVIDER):\n")
    for row in rows:
        mark = "✓" if row.ready else "✗"
        arrow = "→" if row.selected else " "
        print(f" {arrow} {mark} {row.name:10} {row.message}")
    selected = next((r for r in rows if r.selected), None)
    if selected is None:
        print(f"\nDOCSHELP_PROVIDER is not one of: {', '.join(r.name for r in rows)}")
        return 1
    print(f"\nSelected: {selected.display_name}. {'Ready to run the stages.' if selected.ready else 'Fix the issue above first.'}")
    return 0 if selected.ready else 1


def doctor_cli() -> None:
    sys.exit(doctor_main())


# --- lessons: lessons/p<part>_<topic>/<NN>_<topic>.py, addressed as "<part>.<NN>" ------------------

LESSONS_DIR = PROJECT_ROOT / "lessons"


def find_lessons(root: Path = LESSONS_DIR) -> list[tuple[str, Path]]:
    """Every lesson as ("<part>.<number>", path), in curriculum order."""
    found = []
    for path in root.glob("p*_*/[0-9][0-9]_*.py"):
        part = int(path.parent.name[1:].split("_", 1)[0])
        found.append(((part, int(path.name[:2])), path))
    return [(f"{part}.{number}", path) for (part, number), path in sorted(found)]


def lesson_path(code: str) -> Path:
    """Path of lesson "1.6" (also accepts "01.06"). Raises KeyError if there is no such lesson."""
    try:
        wanted = tuple(int(n) for n in code.split("."))
    except ValueError:
        raise KeyError(code) from None
    for found, path in find_lessons():
        if tuple(map(int, found.split("."))) == wanted:
            return path
    raise KeyError(code)


def lesson_title(path: Path) -> str:
    """First line of the lesson's docstring."""
    docstring = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
    return docstring.splitlines()[0] if docstring else ""


def load_lesson(code: str) -> ModuleType:
    """Import a lesson as a module without running `main()` (file names start with digits, so no `import`)."""
    path = lesson_path(code)
    spec = importlib.util.spec_from_file_location(f"lesson_{code.replace('.', '_')}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lessons_main() -> None:
    ensure_utf8()
    part = None
    for code, path in find_lessons():
        if path.parent.name != part:
            part = path.parent.name
            print(f"\n{part}")
        print(f"  {code:6} {lesson_title(path)}")
    print("\nRun one with: uv run poe lesson <number>   e.g. uv run poe lesson 1.6")


def lesson_main(argv: list[str] | None = None) -> None:
    """`uv run poe lesson 1.6 [args...]`: run that lesson's script as `__main__`, passing the extra args on."""
    ensure_utf8()
    argv = sys.argv[1:] if argv is None else argv
    try:
        path = lesson_path(argv[0]) if argv else None
    except KeyError:
        path = None
    if path is None:
        codes = ", ".join(code for code, _ in find_lessons())
        print(f"Usage: uv run poe lesson <number> [args]. Lessons: {codes}", file=sys.stderr)
        sys.exit(1)
    sys.argv = [str(path), *argv[1:]]
    runpy.run_path(str(path), run_name="__main__")
