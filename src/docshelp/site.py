"""The project website: the repo's Markdown, plus a page per part built from the lesson docstrings (GitHub Pages).

`uv run poe site` generates `.site-src/` (MkDocs sources and `mkdocs.yml`) and builds the static site into
`.site/`; `uv run poe site-serve` previews it at http://127.0.0.1:8000. CI builds it on every PR and
`.github/workflows/pages.yml` publishes it from `main`.

Nothing is written twice. The generator reads what the repo already has:
  * README.md -> the home page; CURRICULUM.md, ROADMAP.md, MAINTENANCE.md, AGENTS.md -> pages
  * each lesson's docstring (title, Academy lessons, concepts, run command) -> lessons/part-N.md
  * links between those files become site links; links to code go to the file on GitHub
  * `<details>` self-check questions become collapsible question boxes
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = "https://github.com/RGB314/docshelp-langchain-curriculum"
SITE_URL = "https://rgb314.github.io/docshelp-langchain-curriculum/"
SOURCE_DIR = ROOT / ".site-src"
BUILD_DIR = ROOT / ".site"

PAGES = {  # repo file -> site page
    "README.md": "index.md",
    "CURRICULUM.md": "curriculum.md",
    "ROADMAP.md": "roadmap.md",
    "MAINTENANCE.md": "maintenance.md",
    "AGENTS.md": "contributing.md",
}
LESSON_PATH = re.compile(r"lessons/p(\d+)_[\w]+/(\d\d)_[\w]+\.py")


# --- lessons -----------------------------------------------------------------------------------------------


def lesson_files(root: Path = ROOT) -> dict[int, list[Path]]:
    """Lesson files by part number, in order."""
    parts: dict[int, list[Path]] = {}
    for path in sorted(root.glob("lessons/p*_*/[0-9][0-9]_*.py")):
        parts.setdefault(int(path.parent.name[1:].split("_", 1)[0]), []).append(path)
    return dict(sorted(parts.items()))


def anchor(part: int, number: int) -> str:
    return f"lesson-{part}-{number}"


def part_titles(curriculum: str) -> dict[int, tuple[str, str]]:
    """Part number -> (name, topics) from CURRICULUM.md's "The nine parts" table."""
    titles = {}
    for number, theme in re.findall(r"^\| (\d) \| ([^|]+) \|", curriculum, re.MULTILINE):
        name, _, topics = theme.partition(":")
        titles[int(number)] = (name.strip(), topics.strip())
    return titles


def docstring_markdown(docstring: str) -> str:
    """A lesson docstring (minus its title line) as Markdown: labels in bold, the run commands as a code block."""
    lines = docstring.splitlines()[1:]
    out: list[str] = []
    run_block: list[str] = []
    for line in lines:
        if run_block or line.startswith("Run:"):
            run_block.append(line.removeprefix("Run:").strip())
            continue
        label = re.match(r"^([A-Z][\w ()/,.'-]{1,60}):(\s|$)(.*)", line)
        if label and not line.startswith(" "):
            rest = label.group(3).strip()
            # "Academy: ..." keeps its text in the paragraph; "Concepts:" alone is followed by a list
            out += ["", f"**{label.group(1)}:** {rest}".rstrip()] + ([] if rest else [""])
        else:
            out.append(line)
    text = "\n".join(out).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if run_block:
        commands = "\n".join(shell_comment(c) for c in run_block if c)
        text += f"\n\n**Run:**\n\n```bash\n{commands}\n```"
    return text


def shell_comment(line: str) -> str:
    """`uv run poe server   (terminal 1)` -> `uv run poe server   # terminal 1`, so the block is valid shell."""
    match = re.match(r"^(.*?\S)(\s{2,})\((.*)\)$", line)
    return f"{match.group(1)}{match.group(2)}# {match.group(3)}" if match else line


def lesson_page(part: int, paths: list[Path], titles: dict[int, tuple[str, str]], root: Path = ROOT) -> str:
    name, topics = titles.get(part, (f"Part {part}", ""))
    sections, index = [], []
    for path in paths:
        number = int(path.name[:2])
        docstring = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
        title = (docstring.splitlines()[0] if docstring else path.stem).rstrip(".")
        code = f"{part}.{number}"
        index.append(f"| [{code}](#{anchor(part, number)}) | {title} |")
        source = f"{REPO}/blob/main/{path.relative_to(root).as_posix()}"
        sections.append(
            f"## {code} {title} {{#{anchor(part, number)}}}\n\n{docstring_markdown(docstring)}\n\n"
            f"[Read the lesson's code on GitHub]({source})"
        )
    header = f"# Part {part}: {name}\n\n{topics[:1].upper() + topics[1:]}.\n\n" if topics else f"# Part {part}: {name}\n\n"
    table = "| Lesson | Title |\n|---|---|\n" + "\n".join(index)
    return header + table + "\n\n" + "\n\n".join(sections) + "\n"


# --- the repo's Markdown, adapted for the site ---------------------------------------------------------------


def github_url(path: str, root: Path = ROOT) -> str:
    kind = "tree" if path.endswith("/") or (root / path).is_dir() else "blob"
    return f"{REPO}/{kind}/main/{path.rstrip('/')}"


def site_target(target: str, page_dir: str = "") -> str:
    """Where a repo-relative link should point on the site (`page_dir`: the linking page's folder)."""
    path, hash_mark, fragment = target.partition("#")
    prefix = "../" * page_dir.count("/") if page_dir else ""
    if path in PAGES:
        return prefix + PAGES[path] + (hash_mark + fragment if fragment else "")
    lesson = LESSON_PATH.fullmatch(path)
    if lesson:
        part, number = int(lesson.group(1)), int(lesson.group(2))
        return f"{prefix}lessons/part-{part}.md#{anchor(part, number)}"
    return github_url(path) + (hash_mark + fragment if fragment else "")


def rewrite_links(markdown: str, page_dir: str = "") -> str:
    def link(match: re.Match) -> str:
        text, target = match.group(1), match.group(2)
        if "://" in target or target.startswith(("#", "mailto:")):
            return match.group(0)
        return f"[{text}]({site_target(target, page_dir)})"

    markdown = re.sub(r"\[([^\]]*)\]\(([^)\s]+)\)", link, markdown)
    # Paths written as code (`lessons/p1_langchain/06_tools.py`, `src/docshelp/deep.py`) become links too.
    def code_path(match: re.Match) -> str:
        path = match.group(1)
        if not (ROOT / path.rstrip("/")).exists():  # e.g. generated, git-ignored files: nothing to link to
            return match.group(0)
        return f"[`{path}`]({site_target(path, page_dir)})"

    return re.sub(r"(?<!\[)`((?:lessons|src|tests|data|skills|agents|evals)/[\w./-]+)`(?!\])", code_path, markdown)


def details_to_admonitions(markdown: str) -> str:
    """`<details><summary>Q</summary> A </details>` -> a collapsible Material "question" box."""

    def box(match: re.Match) -> str:
        summary = re.sub(r"<code>(.*?)</code>", r"`\1`", match.group(1)).replace('"', "'")
        body = "\n".join(f"    {line}" if line.strip() else "" for line in match.group(2).strip().splitlines())
        return f'??? question "{summary}"\n\n{body}\n'

    return re.sub(r"<details>\s*<summary>(.*?)</summary>\s*(.*?)\s*</details>", box, markdown, flags=re.DOTALL)


def without_contents(markdown: str) -> str:
    """Drop the README's hand-made table of contents: the site has navigation and a page outline."""
    return re.sub(r"^## Contents\n.*?(?=^## )", "", markdown, flags=re.DOTALL | re.MULTILINE)


def page_from(name: str, root: Path = ROOT) -> str:
    text = (root / name).read_text(encoding="utf-8")
    if name == "README.md":
        text = without_contents(text)
    return details_to_admonitions(rewrite_links(text))


# --- the MkDocs project ------------------------------------------------------------------------------------


def mkdocs_config(parts: dict[int, tuple[str, str]], lesson_parts: list[int]) -> str:
    lesson_nav = "\n".join(
        f"      - 'Part {p}: {parts.get(p, (f'Part {p}', ''))[0]}': lessons/part-{p}.md" for p in lesson_parts
    )
    return f"""# Generated by `uv run poe site` (src/docshelp/site.py). Edit the generator, not this file.
site_name: DocsHelp, the LangChain curriculum
site_description: Every LangChain Academy course, applied to one support assistant, at zero cost.
site_url: {SITE_URL}
repo_url: {REPO}
repo_name: RGB314/docshelp-langchain-curriculum
edit_uri: ""
docs_dir: docs
site_dir: ../.site
theme:
  name: material
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.top
    - navigation.instant
    - search.highlight
    - content.code.copy
  palette:
    - media: "(prefers-color-scheme: light)"
      scheme: default
      primary: indigo
      accent: teal
      toggle: {{icon: material/weather-night, name: Dark mode}}
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      primary: indigo
      accent: teal
      toggle: {{icon: material/weather-sunny, name: Light mode}}
  icon:
    repo: fontawesome/brands/github
markdown_extensions:
  - admonition
  - attr_list
  - md_in_html
  - tables
  - toc: {{permalink: true}}
  - pymdownx.details
  - pymdownx.highlight
  - pymdownx.inlinehilite
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
nav:
  - Home: index.md
  - Curriculum: curriculum.md
  - Lessons:
{lesson_nav}
  - Project:
      - Roadmap: roadmap.md
      - Maintenance: maintenance.md
      - Contributing: contributing.md
"""


def generate(out: Path = SOURCE_DIR, root: Path = ROOT) -> Path:
    """Write the MkDocs sources and config into `out`; returns the config path."""
    if out.exists():
        shutil.rmtree(out)
    docs = out / "docs"
    (docs / "lessons").mkdir(parents=True)
    for repo_file, page in PAGES.items():
        if (root / repo_file).exists():
            (docs / page).write_text(page_from(repo_file, root), encoding="utf-8")
    titles = part_titles((root / "CURRICULUM.md").read_text(encoding="utf-8"))
    lessons = lesson_files(root)
    for part, paths in lessons.items():
        page = rewrite_links(lesson_page(part, paths, titles, root), page_dir="lessons/")
        (docs / "lessons" / f"part-{part}.md").write_text(page, encoding="utf-8")
    config = out / "mkdocs.yml"
    config.write_text(mkdocs_config(titles, list(lessons)), encoding="utf-8")
    return config


def mkdocs(*args: str) -> int:
    return subprocess.run([sys.executable, "-m", "mkdocs", *args]).returncode


def build_main() -> None:
    config = generate()
    code = mkdocs("build", "--strict", "-f", str(config))
    if code == 0:
        print(f"Built the site into {BUILD_DIR.relative_to(ROOT).as_posix()}/ (open index.html, or `uv run poe site-serve`).")
    sys.exit(code)


def serve_main() -> None:
    sys.exit(mkdocs("serve", "-f", str(generate())))
