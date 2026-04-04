FROM python:3.10-slim

# Install uv (required by openenv validator)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# Copy dependency spec first for layer caching
COPY --chown=user pyproject.toml uv.lock* ./

# Install all dependencies (without project itself)
RUN uv sync --no-install-project

# Copy full source code
COPY --chown=user . .

# THE KEY FIX: ensure /app is always in Python's module search path
# so `from server.xxx import ...` works regardless of install state
ENV PYTHONPATH="/app"

# Install the project itself (registers the `server` entry point)
RUN uv sync

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Start via the [project.scripts] entry point: server = "server.__main__:main"
CMD ["uv", "run", "server"]
