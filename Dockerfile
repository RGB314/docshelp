# DocsHelp in a container: the same Linux + Python 3.12 environment on Windows, macOS and Linux hosts.
#   docker compose build
#   docker compose run --rm app poe stage1
# See README "Option B: Docker".

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONUTF8=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# 1) Dependencies only, exactly as pinned in uv.lock. This layer is rebuilt only when the lockfile changes.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-extras --no-install-project

# 2) The project itself.
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-extras

# 3) Bake the local embedding model into the image so containers start fast and RAG works offline.
RUN uv run --no-sync python -c "from docshelp.config import get_embeddings; get_embeddings()"

# Use the project's virtual environment directly: `poe`, `python`, `pytest`, `langgraph` are on PATH.
ENV PATH="/app/.venv/bin:$PATH" \
    UV_NO_SYNC=1

EXPOSE 2024
CMD ["poe"]
