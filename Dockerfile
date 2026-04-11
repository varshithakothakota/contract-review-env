FROM python:3.10-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /app
COPY --chown=user pyproject.toml uv.lock* ./
RUN uv sync --no-install-project
COPY --chown=user . .
ENV PYTHONPATH="/app"
RUN uv sync
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"
CMD ["uv", "run", "server"]
