"""`uv run poe maintenance`: a periodic health report for the curriculum (see MAINTENANCE.md).

It checks the things that drift while nobody is looking, and writes `data/reports/maintenance.md`:

  1. the Academy catalogue: course links on the catalogue page that `data/academy_catalogue.json` doesn't know
     yet (new courses). The page renders most of its list with JavaScript, so this catches what the static
     HTML shows; MAINTENANCE.md describes the full check, course by course.
  2. dependencies: what `uv lock --upgrade` would update (dry run: nothing changes).
  3. CI hangs: stack dumps from pytest's `faulthandler_timeout` (pyproject.toml) in recent failed or cancelled
     CI runs, with the test that was stuck. Needs the GitHub CLI (`gh`), signed in.
  4. the curriculum itself: every lesson file mapped in CURRICULUM.md, no "(planned)" rows.

Every check is best-effort: a missing network, `gh` or `uv` is reported, never fatal.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import urllib.request
from collections.abc import Callable
from datetime import date
from pathlib import Path

from docshelp.config import DATA_DIR, PROJECT_ROOT
from docshelp.console import ensure_utf8

CATALOGUE_URL = "https://academy.langchain.com/collections"
CATALOGUE_PATH = DATA_DIR / "academy_catalogue.json"
REPORT_PATH = DATA_DIR / "reports" / "maintenance.md"
HANG_MARKER = re.compile(r"Timeout \(\d+(?::\d{2}){2}\)!")  # "Timeout (0:05:00)!"
STUCK_TEST = re.compile(r'File "[^"]*/(tests/[^"]+\.py)", line \d+ in (test_\w+)')


def run(command: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        command, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout, stdin=subprocess.DEVNULL,
    )


# --- 1. the Academy catalogue ------------------------------------------------------------------------------


def course_slugs(html: str) -> list[str]:
    """Course slugs linked from the catalogue page, e.g. "intro-to-langgraph"."""
    return sorted(set(re.findall(r'href="(?:https://academy\.langchain\.com)?/courses/([\w-]+)', html)))


def new_courses(html: str, known: dict[str, str]) -> list[str]:
    return [slug for slug in course_slugs(html) if slug not in known]


def check_catalogue(fetch: Callable[[str], str] | None = None) -> dict:
    known = json.loads(CATALOGUE_PATH.read_text(encoding="utf-8"))["courses"]
    try:
        html = (fetch or _fetch)(CATALOGUE_URL)
    except OSError as err:
        return {"ok": False, "note": f"couldn't fetch the catalogue ({err})", "new": [], "seen": 0}
    seen = course_slugs(html)
    return {"ok": True, "note": "", "new": new_courses(html, known), "seen": len(seen)}


def _fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "docshelp-maintenance"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


# --- 2. dependencies ---------------------------------------------------------------------------------------


def parse_updates(output: str) -> list[str]:
    """`uv lock --upgrade --dry-run` lines such as "Update langgraph v1.2.12 -> v1.2.14"."""
    return [line.strip().removeprefix("Update ") for line in output.splitlines() if line.strip().startswith("Update ")]


def check_dependencies() -> dict:
    if shutil.which("uv") is None:
        return {"ok": False, "note": "uv is not on PATH", "updates": []}
    result = run(["uv", "lock", "--upgrade", "--dry-run"])
    return {"ok": result.returncode == 0, "note": result.stderr[-300:] if result.returncode else "",
            "updates": parse_updates(result.stdout + result.stderr)}


# --- 3. CI hangs -------------------------------------------------------------------------------------------


def hang_dumps(log: str) -> list[str]:
    """The tests a CI log shows stuck: each faulthandler dump, and the first test frame after it."""
    stuck = []
    lines = log.splitlines()
    for i, line in enumerate(lines):
        if HANG_MARKER.search(line):
            for later in lines[i : i + 400]:
                match = STUCK_TEST.search(later)
                if match:
                    stuck.append(f"{match.group(1)}::{match.group(2)}")
                    break
            else:
                stuck.append("(stack dump without a test frame)")
    return stuck


def check_ci(runs: int = 20) -> dict:
    if shutil.which("gh") is None:
        return {"ok": False, "note": "the GitHub CLI (gh) is not installed", "hangs": []}
    listing = run(["gh", "run", "list", "--limit", str(runs), "--json", "databaseId,conclusion,headBranch,createdAt,url"])
    if listing.returncode != 0:
        return {"ok": False, "note": f"gh run list failed: {listing.stderr.strip()[:200]}", "hangs": []}
    hangs = []
    for ci_run in json.loads(listing.stdout or "[]"):
        if ci_run["conclusion"] not in ("failure", "cancelled", "timed_out"):
            continue
        log = run(["gh", "run", "view", str(ci_run["databaseId"]), "--log"], timeout=600)
        for test in hang_dumps(log.stdout):
            hangs.append({"run": ci_run["url"], "branch": ci_run["headBranch"], "date": ci_run["createdAt"][:10],
                          "test": test})
    return {"ok": True, "note": "", "hangs": hangs}


# --- 4. the curriculum -------------------------------------------------------------------------------------


def check_curriculum() -> dict:
    from docshelp.tasks import find_lessons

    text = (PROJECT_ROOT / "CURRICULUM.md").read_text(encoding="utf-8")
    unmapped = [code for code, path in find_lessons() if path.relative_to(PROJECT_ROOT).as_posix() not in text]
    return {"ok": not unmapped and "(planned)" not in text, "unmapped": unmapped, "planned": text.count("(planned)")}


# --- the report --------------------------------------------------------------------------------------------


def report(catalogue: dict, dependencies: dict, ci: dict, curriculum: dict) -> str:
    lines = [f"# Maintenance report, {date.today().isoformat()}", "", "## 1. Academy catalogue", ""]
    if not catalogue["ok"]:
        lines.append(f"Not checked: {catalogue['note']}.")
    elif catalogue["new"]:
        lines += [f"New course links on the catalogue page ({catalogue['seen']} seen):", ""]
        lines += [f"- https://academy.langchain.com/courses/{slug}" for slug in catalogue["new"]]
        lines += ["", "Map each one in CURRICULUM.md, then add it to data/academy_catalogue.json (MAINTENANCE.md)."]
    else:
        lines.append(f"No unknown course links ({catalogue['seen']} on the page). Still do the course-by-course check.")
    lines += ["", "## 2. Dependency updates (dry run)", ""]
    if not dependencies["ok"]:
        lines.append(f"Not checked: {dependencies['note']}")
    else:
        lines += [f"- {u}" for u in dependencies["updates"]] or ["Everything is at the newest allowed version."]
    lines += ["", "## 3. CI hangs (faulthandler stack dumps)", ""]
    if not ci["ok"]:
        lines.append(f"Not checked: {ci['note']}.")
    else:
        lines += [f"- {h['date']} `{h['branch']}`: `{h['test']}` ({h['run']})" for h in ci["hangs"]] or [
            "No stack dumps in recent failed or cancelled runs."
        ]
    lines += ["", "## 4. Curriculum consistency", ""]
    lines.append("OK: every lesson is mapped, no planned rows." if curriculum["ok"] else
                 f"Unmapped lessons: {curriculum['unmapped']}; planned rows: {curriculum['planned']}.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ensure_utf8()
    args = sys.argv[1:] if argv is None else argv
    checks = {
        "catalogue": check_catalogue,
        "dependencies": check_dependencies,
        "ci": (lambda: {"ok": False, "note": "skipped (--no-ci)", "hangs": []}) if "--no-ci" in args else check_ci,
        "curriculum": check_curriculum,
    }
    results = {}
    for name, check in checks.items():
        print(f"Checking {name}...", flush=True)
        results[name] = check()
    text = report(results["catalogue"], results["dependencies"], results["ci"], results["curriculum"])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")
    shown = REPORT_PATH.relative_to(PROJECT_ROOT) if REPORT_PATH.is_relative_to(PROJECT_ROOT) else REPORT_PATH
    print(f"\n{text}\nSaved to {shown.as_posix()}")
