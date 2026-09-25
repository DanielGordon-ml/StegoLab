"""Https transport behaviour checked against a loopback server without TLS."""

import socket

import pytest
from dataset_source_fixtures import (
    LoopbackServer,
    Response,
    resolver_answering,
    transport_for,
)

from backend_service.failures import ApplicationFailure

USER_AGENT = "stegolab-dataset-fetch/0.1"


def test_body_with_content_length() -> None:
    """A plain 200 exposes lower-cased headers and streams the whole body."""
    routes = {
        ("GET", "/file"): Response(
            body=b"x" * 3000, headers={"Content-Type": "application/octet-stream"}
        )
    }
    with LoopbackServer(routes) as server:
        with transport_for(server).open("GET", server.url("/file")) as response:
            assert response.status == 200
            assert response.headers["content-length"] == "3000"
            assert response.headers["content-type"] == "application/octet-stream"
            assert response.final_url == server.url("/file")
            assert b"".join(response.iter_bytes(1024)) == b"x" * 3000
        request = server.requests[0]
        assert request.headers["host"] == f"127.0.0.1:{server.port}"
        assert request.headers["user-agent"] == USER_AGENT
        assert request.headers["accept-encoding"] == "identity"
        assert "authorization" not in request.headers


def test_chunked_body_without_content_length() -> None:
    """A chunked body streams to its end although no length was declared."""
    routes = {("GET", "/stream"): Response(body=b"chunked-data" * 500, chunked=True)}
    with LoopbackServer(routes) as server:
        with transport_for(server).open("GET", server.url("/stream")) as response:
            assert "content-length" not in response.headers
            assert b"".join(response.iter_bytes()) == b"chunked-data" * 500


def test_head_returns_headers_only() -> None:
    """HEAD carries the headers of the file and an empty body."""
    routes = {("HEAD", "/file"): Response(headers={"ETag": '"abc"'}, body=b"y" * 10)}
    with LoopbackServer(routes) as server:
        with transport_for(server).open("HEAD", server.url("/file")) as response:
            assert response.status == 200
            assert response.headers["etag"] == '"abc"'
            assert response.headers["content-length"] == "10"
            assert list(response.iter_bytes()) == []
        assert server.requests[0].method == "HEAD"


def test_other_statuses_are_returned_as_they_are() -> None:
    """Non-redirect statuses are handed back for the caller to judge."""
    routes = {("GET", "/missing"): Response(status=404, body=b"gone")}
    with LoopbackServer(routes) as server:
        with transport_for(server).open("GET", server.url("/missing")) as response:
            assert response.status == 404
            assert b"".join(response.iter_bytes()) == b"gone"


def test_redirect_chain_ends_at_the_final_url() -> None:
    """Relative and absolute redirects are followed and every hop is logged."""
    routes: dict[tuple[str, str], Response] = {}
    with LoopbackServer(routes) as server:
        routes[("GET", "/a")] = Response(302, {"Location": "/b"})
        routes[("GET", "/b")] = Response(307, {"Location": server.url("/c")})
        routes[("GET", "/c")] = Response(body=b"done")
        with transport_for(server).open("GET", server.url("/a")) as response:
            assert response.status == 200
            assert response.final_url == server.url("/c")
            assert b"".join(response.iter_bytes()) == b"done"
        assert [request.path for request in server.requests] == ["/a", "/b", "/c"]


def test_redirect_limit() -> None:
    """Six redirects exceed the default limit of five and stop the download."""
    routes = {
        ("GET", f"/hop{number}"): Response(301, {"Location": f"/hop{number + 1}"})
        for number in range(1, 7)
    }
    routes[("GET", "/hop7")] = Response(body=b"never")
    with LoopbackServer(routes) as server:
        with pytest.raises(ApplicationFailure) as failure:
            transport_for(server).open("GET", server.url("/hop1"))
        assert failure.value.code == "download_redirect_limit"
        assert len(server.requests) == 6


def test_redirect_without_location_is_refused() -> None:
    """A redirect that names no destination is treated as a bad address."""
    routes = {("GET", "/a"): Response(303)}
    with LoopbackServer(routes) as server:
        with pytest.raises(ApplicationFailure) as failure:
            transport_for(server).open("GET", server.url("/a"))
        assert failure.value.code == "download_address"


def test_redirect_to_private_address_is_refused() -> None:
    """A redirect into a private network fails before any connection."""
    routes = {("GET", "/a"): Response(302, {"Location": "https://10.0.0.1/secret"})}
    with LoopbackServer(routes) as server:
        with pytest.raises(ApplicationFailure) as failure:
            transport_for(server).open("GET", server.url("/a"))
        assert failure.value.code == "download_address_blocked"
        assert "10.0.0.1" not in failure.value.message
        assert len(server.requests) == 1


def test_authorization_only_for_the_exact_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The header goes to the registered host and not to a redirect target."""
    routes: dict[tuple[str, str], Response] = {}
    with LoopbackServer(routes) as server:
        monkeypatch.setattr(
            socket, "getaddrinfo", resolver_answering({"localhost": ["127.0.0.1"]})
        )
        routes[("GET", "/private")] = Response(
            302, {"Location": server.url("/public", host="localhost")}
        )
        routes[("GET", "/public")] = Response(body=b"ok")
        transport = transport_for(
            server, authorization={"127.0.0.1": "Bearer secret-token"}
        )
        with transport.open("GET", server.url("/private")) as response:
            assert response.status == 200
        first, second = server.requests
        assert first.headers["authorization"] == "Bearer secret-token"
        assert second.path == "/public"
        assert second.headers["host"] == f"localhost:{server.port}"
        assert "authorization" not in second.headers
        assert "secret-token" not in repr(transport)
        assert "127.0.0.1" in repr(transport)


def test_range_headers_are_forwarded() -> None:
    """Range and If-Range reach the server; caller headers cannot add credentials."""
    routes = {("GET", "/file"): Response(body=b"0123456789")}
    with LoopbackServer(routes) as server:
        transport = transport_for(server)
        with transport.open(
            "GET",
            server.url("/file"),
            byte_range=(4, None),
            if_range='"tag"',
            headers={"Authorization": "Bearer leaked", "X-Extra": "1"},
        ):
            pass
        with transport.open("GET", server.url("/file"), byte_range=(0, 99)):
            pass
        first, second = server.requests
        assert first.headers["range"] == "bytes=4-"
        assert first.headers["if-range"] == '"tag"'
        assert first.headers["x-extra"] == "1"
        assert "authorization" not in first.headers
        assert second.headers["range"] == "bytes=0-99"
        assert "if-range" not in second.headers


def test_connection_refused() -> None:
    """A closed port becomes the fixed connection message."""
    with LoopbackServer({}) as server:
        transport = transport_for(server)
    with pytest.raises(ApplicationFailure) as failure:
        transport.open("GET", server.url("/file"))
    assert failure.value.code == "download_connection"
    assert str(server.port) not in failure.value.message
