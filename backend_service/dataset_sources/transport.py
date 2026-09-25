"""Https transport that pins each connection to a checked address."""

import http.client
import logging
import socket
import ssl
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from types import TracebackType
from typing import Literal
from urllib.parse import urljoin

from backend_service.dataset_sources.address_policy import (
    AddressPolicy,
    ValidatedUrl,
    address_failure,
    resolve_and_check,
    validate_url,
)
from backend_service.failures import ApplicationFailure

logger = logging.getLogger(__name__)

Connector = Callable[[str, str, int, float], socket.socket]
USER_AGENT = "stegolab-dataset-fetch/0.1"
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
MAXIMUM_LOCATION_LENGTH = 4096
RESERVED_HEADERS = frozenset(
    {"host", "authorization", "user-agent", "accept-encoding", "range", "if-range"}
)
_MESSAGES = {
    "download_connection": (
        "The download connection failed. Check the address and try again."
    ),
    "download_redirect_limit": "The download was redirected too many times.",
    "download_stalled": "The download stopped receiving data.",
}


def transport_failure(code: str) -> ApplicationFailure:
    """Return the fixed message for a transport failure code."""
    return ApplicationFailure(code, _MESSAGES[code], 502)


def secure_connector(ssl_context: ssl.SSLContext | None = None) -> Connector:
    """Connect by TCP to the checked address, then verify TLS for the host name.

    The socket reaches the resolved address; the certificate must match the name.
    """
    context = ssl_context or ssl.create_default_context()

    def connect(address: str, host: str, port: int, timeout: float) -> socket.socket:
        """Open a verified TLS socket to one already checked address."""
        plain = socket.create_connection((address, port), timeout=timeout)
        try:
            return context.wrap_socket(plain, server_hostname=host)
        except OSError:
            plain.close()
            raise

    return connect


@dataclass
class TransportResponse:
    """One http response with lower-cased headers and a streaming body."""

    status: int
    headers: dict[str, str]
    final_url: str
    _response: http.client.HTTPResponse = field(repr=False)
    _connection: http.client.HTTPConnection = field(repr=False)
    _socket: socket.socket = field(repr=False)

    def set_read_timeout(self, seconds: float) -> None:
        """Limit how long one body read may wait before it counts as a stall."""
        self._socket.settimeout(seconds)

    def iter_bytes(self, chunk_size: int = 1_048_576) -> Iterator[bytes]:
        """Yield body pieces as they arrive without waiting for a full chunk."""
        try:
            while True:
                chunk = self._response.read1(chunk_size)
                if not chunk:
                    return
                yield chunk
        except TimeoutError:
            raise transport_failure("download_stalled") from None
        except (OSError, http.client.HTTPException):
            raise transport_failure("download_connection") from None

    def close(self) -> None:
        """Release the response, the connection and the socket beneath them."""
        self._response.close()
        self._connection.close()
        self._socket.close()

    def __enter__(self) -> "TransportResponse":
        """Hand the response to a ``with`` block."""
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the response when the block ends."""
        self.close()


class SecureTransport:
    """Open GET and HEAD requests through checked addresses, following redirects."""

    def __init__(
        self,
        policy: AddressPolicy,
        *,
        authorization: dict[str, str] | None = None,
        connector: Connector | None = None,
        connect_timeout: float = 20.0,
        read_timeout: float = 60.0,
        maximum_redirects: int = 5,
    ) -> None:
        """Keep the policy, exact-host authorization values and timeouts."""
        self.policy = policy
        self._authorization = dict(authorization or {})
        self._connector = connector or secure_connector()
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.maximum_redirects = maximum_redirects

    def __repr__(self) -> str:
        """Describe the transport without any header values."""
        return (
            f"SecureTransport(policy={self.policy!r}, "
            f"authorization_hosts={sorted(self._authorization)!r}, "
            f"maximum_redirects={self.maximum_redirects})"
        )

    def open(
        self,
        method: Literal["GET", "HEAD"],
        url: str,
        *,
        headers: dict[str, str] | None = None,
        byte_range: tuple[int, int | None] | None = None,
        if_range: str | None = None,
    ) -> TransportResponse:
        """Send one request and return the first response that is not a redirect."""
        current_url = url
        for _hop in range(self.maximum_redirects + 1):
            target = validate_url(current_url, self.policy)
            address = resolve_and_check(target.host, self.policy)[0]
            connection, response, stream_socket = self._send(
                target, address, method, headers, byte_range, if_range
            )
            if response.status not in REDIRECT_STATUSES:
                logger.info("download_response_received")
                return TransportResponse(
                    status=response.status,
                    headers={
                        name.lower(): value for name, value in response.getheaders()
                    },
                    final_url=current_url,
                    _response=response,
                    _connection=connection,
                    _socket=stream_socket,
                )
            location = response.getheader("Location")
            response.close()
            connection.close()
            stream_socket.close()
            if not location or len(location) > MAXIMUM_LOCATION_LENGTH:
                raise address_failure("download_address")
            logger.info("download_redirect_followed")
            current_url = urljoin(current_url, location)
        raise transport_failure("download_redirect_limit")

    def _send(
        self,
        target: ValidatedUrl,
        address: str,
        method: Literal["GET", "HEAD"],
        headers: dict[str, str] | None,
        byte_range: tuple[int, int | None] | None,
        if_range: str | None,
    ) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse, socket.socket]:
        """Connect to one checked address, send the headers and keep the socket."""
        request_headers = {
            name: value
            for name, value in (headers or {}).items()
            if name.lower() not in RESERVED_HEADERS
        }
        request_headers["Host"] = _host_header(target)
        request_headers["User-Agent"] = USER_AGENT
        request_headers["Accept-Encoding"] = "identity"
        if byte_range is not None:
            start, end = byte_range
            request_headers["Range"] = f"bytes={start}-{'' if end is None else end}"
        if if_range is not None:
            request_headers["If-Range"] = if_range
        authorization = self._authorization.get(target.host)
        if authorization is not None:
            request_headers["Authorization"] = authorization
        connection = http.client.HTTPConnection(
            target.host, target.port, timeout=self.read_timeout
        )
        stream_socket: socket.socket | None = None
        try:
            stream_socket = self._connector(
                address, target.host, target.port, self.connect_timeout
            )
            stream_socket.settimeout(self.read_timeout)
            connection.sock = stream_socket
            connection.request(method, target.path_and_query, headers=request_headers)
            return connection, connection.getresponse(), stream_socket
        except (OSError, http.client.HTTPException, ValueError):
            connection.close()
            if stream_socket is not None:
                stream_socket.close()
            logger.warning("download_connection_failed")
            raise transport_failure("download_connection") from None


def _host_header(target: ValidatedUrl) -> str:
    """Format the Host header, bracketing IPv6 literals and showing odd ports."""
    host = f"[{target.host}]" if ":" in target.host else target.host
    return host if target.port == 443 else f"{host}:{target.port}"
