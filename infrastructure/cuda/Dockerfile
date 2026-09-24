# Same pinned Python and uv bases as the CPU application; build on native amd64.
FROM ghcr.io/astral-sh/uv:0.10.12@sha256:72ab0aeb448090480ccabb99fb5f52b0dc3c71923bffb5e2e26517a1c27b7fec AS dependency_tool
FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e AS cuda_runtime

COPY --from=dependency_tool /uv /usr/local/bin/uv
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PYTHONPATH=/application \
    PATH="/application/.venv/bin:$PATH"
WORKDIR /application
RUN test "$(uname -m)" = "x86_64"
COPY infrastructure/cuda/pyproject.toml infrastructure/cuda/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --no-cache \
    && groupadd --gid 10001 stegolab \
    && useradd --uid 10001 --gid stegolab --no-create-home stegolab \
    && mkdir -p /data/state /data/logs /data/checkpoints /data/models \
    && chown -R 10001:10001 /data
COPY backend_service/ ./backend_service/
COPY schemas/ ./schemas/
COPY scripts/ ./scripts/
COPY infrastructure/cuda/runtime_dependencies.json ./infrastructure/cuda/runtime_dependencies.json
USER 10001:10001
EXPOSE 8000
CMD ["uvicorn", "backend_service.application:create_application", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
