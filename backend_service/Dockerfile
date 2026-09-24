# Multi-platform image digests verified on 2026-09-23.
FROM ghcr.io/astral-sh/uv:0.10.12@sha256:72ab0aeb448090480ccabb99fb5f52b0dc3c71923bffb5e2e26517a1c27b7fec AS dependency_tool
FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e

COPY --from=dependency_tool /uv /usr/local/bin/uv
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/application/.venv/bin:$PATH"
WORKDIR /application
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-default-groups --no-install-project \
    && groupadd --gid 10001 stegolab \
    && useradd --uid 10001 --gid stegolab --no-create-home stegolab \
    && mkdir -p /data/state /data/logs \
    && chown -R 10001:10001 /data
COPY backend_service/ ./backend_service/
COPY schemas/ ./schemas/
COPY infrastructure/cuda/runtime_dependencies.json ./infrastructure/cuda/runtime_dependencies.json
USER 10001:10001
EXPOSE 8000
CMD ["uvicorn", "backend_service.application:create_application", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
