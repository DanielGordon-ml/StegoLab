"""Loopback http server and helpers for dataset download tests without TLS."""

import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any

from backend_service.dataset_sources.address_policy import AddressPolicy
from backend_service.dataset_sources.transport import SecureTransport

test_policy = AddressPolicy(allow_loopback=True)


@dataclass
class Response:
    """One scripted reply: status, headers, body and optional misbehaviour."""

    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    chunked: bool = False
    slow_chunks: list[tuple[bytes, float]] = field(default_factory=list)
    lie_content_length: int | None = None
    ranges: bool = False
    etag: str | None = None


@dataclass(frozen=True)
class RecordedRequest:
    """What one request looked like when it reached the server."""

    method: str
    path: str
    headers: dict[str, str]


class _Server(ThreadingHTTPServer):
    """Loopback server that keeps the routes and the request log."""

    daemon_threads = True
    block_on_close = False

    def __init__(self, routes: dict[tuple[str, str], Response]) -> None:
        """Bind a random loopback port and remember the scripted routes."""
        super().__init__(("127.0.0.1", 0), _Handler)
        self.routes = routes
        self.requests: list[RecordedRequest] = []
        self.lock = threading.Lock()


class _Handler(BaseHTTPRequestHandler):
    """Answer GET and HEAD from the route table and record each request."""

    protocol_version = "HTTP/1.1"
    server: _Server

    def log_message(self, format: str, *args: object) -> None:
        """Keep the test output quiet."""

    def do_GET(self) -> None:
        """Serve a GET route."""
        self._serve("GET")

    def do_HEAD(self) -> None:
        """Serve a HEAD route without a body."""
        self._serve("HEAD")

    def _serve(self, method: str) -> None:
        """Record the request, then write the scripted reply."""
        headers = {name.lower(): str(value) for name, value in self.headers.items()}
        with self.server.lock:
            self.server.requests.append(RecordedRequest(method, self.path, headers))
        self.close_connection = True
        route = self.server.routes.get((method, self.path))
        if route is None:
            self._reply(404, {"Content-Length": "0"})
            return
        body = route.body or b"".join(piece for piece, _ in route.slow_chunks)
        status, extra = route.status, dict(route.headers)
        if route.etag is not None:
            extra["ETag"] = route.etag
        if route.ranges and headers.get("range", "").startswith("bytes="):
            start = int(headers["range"][6:].split("-", 1)[0])
            if headers.get("if-range") == route.etag:
                extra["Content-Range"] = f"bytes {start}-{len(body) - 1}/{len(body)}"
                status, body = 206, body[start:]
        if route.chunked:
            extra["Transfer-Encoding"] = "chunked"
        elif route.lie_content_length is not None:
            extra["Content-Length"] = str(route.lie_content_length)
        else:
            extra["Content-Length"] = str(len(body))
        self._reply(status, extra)
        if method != "HEAD":
            self._write_body(route, body)

    def _reply(self, status: int, headers: dict[str, str]) -> None:
        """Send the status line and headers, always closing afterwards."""
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header("Connection", "close")
        self.end_headers()

    def _write_body(self, route: Response, body: bytes) -> None:
        """Write the body in one piece or in delayed pieces; ignore lost clients."""
        pieces = route.slow_chunks or [(body, 0.0)]
        try:
            for piece, delay in pieces:
                time.sleep(delay)
                if route.chunked:
                    self.wfile.write(f"{len(piece):x}\r\n".encode() + piece + b"\r\n")
                else:
                    self.wfile.write(piece)
            if route.chunked:
                self.wfile.write(b"0\r\n\r\n")
        except OSError:
            return


class LoopbackServer:
    """Serve scripted routes on 127.0.0.1 for the duration of a ``with`` block."""

    def __init__(self, routes: dict[tuple[str, str], Response]) -> None:
        """Bind immediately so the port is known; serving starts on entry."""
        self._server = _Server(routes)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.02},
            daemon=True,
        )
        self.port = int(self._server.server_address[1])
        self.requests = self._server.requests

    def __enter__(self) -> "LoopbackServer":
        """Start answering requests."""
        self._thread.start()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop the server and release the port."""
        self._server.shutdown()
        self._server.server_close()

    def url(self, path: str, host: str = "127.0.0.1") -> str:
        """Build an https address for this server; the connector skips TLS."""
        return f"https://{host}:{self.port}{path}"


def plain_connector(
    address: str, host: str, port: int, timeout: float
) -> socket.socket:
    """Connect without TLS because the loopback server has no certificate."""
    return socket.create_connection((address, port), timeout=timeout)


def policy_for(server: LoopbackServer) -> AddressPolicy:
    """Allow loopback and the server's random port next to the usual 443."""
    ports = frozenset({443, server.port})
    return AddressPolicy(allow_loopback=True, allowed_ports=ports)


def transport_for(
    server: LoopbackServer, authorization: dict[str, str] | None = None
) -> SecureTransport:
    """Build a transport that reaches the loopback server with short timeouts."""
    return SecureTransport(
        policy_for(server),
        authorization=authorization,
        connector=plain_connector,
        read_timeout=5.0,
    )


def resolver_answering(answers: dict[str, list[str]]) -> Callable[..., list[Any]]:
    """Build a name lookup stand-in that answers listed names and passes on the rest."""
    real = socket.getaddrinfo

    def getaddrinfo(
        host: str | bytes | None, *arguments: Any, **options: Any
    ) -> list[Any]:
        """Return scripted records for known names; delegate address literals."""
        if isinstance(host, str) and host in answers:
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))
                for address in answers[host]
            ]
        return real(host, *arguments, **options)

    return getaddrinfo
