"""Small, shell-independent helpers behind the `uv run poe <task>` commands.

Written in Python rather than PowerShell or bash so they behave identically on Windows, macOS and Linux.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

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
