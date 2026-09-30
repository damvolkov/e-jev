##### BUILDER #####
FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim AS builder
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY README.md ./
COPY src /app/src
RUN uv sync --frozen --no-dev --no-editable

##### DEPLOY #####
FROM python:3.13-slim-trixie AS deploy
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
COPY --from=builder /app/.venv .venv
RUN mkdir -p /data && chown -R nobody /data
USER nobody
EXPOSE 45160
CMD ["granian", "--interface", "asgi", "--factory", "--host", "0.0.0.0", "--port", "45160", "e_jev.main:create_app"]
