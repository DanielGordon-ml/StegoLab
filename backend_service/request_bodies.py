"""Bounded raw request bodies for routes that take a file as the body."""

from collections.abc import Callable

from fastapi import Request

from backend_service.failures import ApplicationFailure


async def read_bounded_body(
    request: Request, limit: int, failure: Callable[[], ApplicationFailure]
) -> bytes:
    """Collect the raw body, refusing it as soon as it passes the byte limit.

    A declared Content-Length above the limit is refused before any byte is
    read. A body without a declared length is counted while it streams and
    refused the moment it passes the limit, so an oversized upload never
    fills memory. The caller supplies the failure so every route keeps its
    own fixed message.
    """
    declared = request.headers.get("content-length")
    if declared is not None and (not declared.isdigit() or int(declared) > limit):
        raise failure()
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise failure()
        chunks.append(chunk)
    return b"".join(chunks)
