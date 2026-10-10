"""Fixtures for the DocsHelp evals (lesson 5.9).

These evals call a real model, so they live outside `tests/` and never run in `uv run poe test`.
Run them with `uv run poe evals` (your configured model) or through lesson 5.9, which can inject a model
via a plugin object (that's how the offline unit test drives them).
"""

import pytest

from docshelp.evals import make_agent_target, make_rag_target


def _injected(request):
    """The plugin object lesson 5.9 passed to `pytest.main(plugins=[...])`, if any."""
    plugins = request.config.pluginmanager.get_plugins()
    return next((p for p in plugins if getattr(p, "provides_docshelp_models", False)), None)


@pytest.fixture(scope="session")
def llm(request):
    plugin = _injected(request)
    if plugin is not None:
        return plugin.llm
    from docshelp.config import ensure_llm_ready, get_llm

    try:
        ensure_llm_ready()
    except SystemExit as err:  # no key / no Ollama: stop the whole run with the setup message
        pytest.exit(str(err), returncode=1)
    return get_llm()


@pytest.fixture(scope="session")
def retriever(request):
    plugin = _injected(request)
    if plugin is not None:
        return plugin.retriever
    from docshelp.ingest import get_retriever

    return get_retriever()


@pytest.fixture(scope="session")
def rag(llm, retriever):
    return make_rag_target(llm, retriever)


@pytest.fixture(scope="session")
def agent(llm, retriever):
    return make_agent_target(llm, retriever)
