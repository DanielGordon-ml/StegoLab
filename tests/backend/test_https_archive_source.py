"""The https archive adapter and the registry that hands out source adapters."""

import socket
import sys
import time
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest
from dataset_source_fixtures import (
    LoopbackServer,
    RecordedRequest,
    Response,
    test_policy,
)

from backend_service.dataset_sources import https_archive, registry
from backend_service.dataset_sources.https_archive import archive_basename, resolve
from backend_service.dataset_sources.plans import PlannedAsset, ResolvedSource
from backend_service.dataset_sources.registry import SourceAdapter, adapter_for
from backend_service.dataset_sources.transport import SecureTransport
from backend_service.failures import ApplicationFailure
from schemas.dataset_sources import (
    DatasetSourceSpec,
    HttpsArchiveSourceSpec,
    HuggingFaceSourceSpec,
    UploadSourceSpec,
)

ARCHIVE_PATH = "/files/DIV2K_valid_HR.zip"
ARCHIVE_URL = f"https://127.0.0.1{ARCHIVE_PATH}"
TERMS = "https://example.test/terms"
LAST_MODIFIED = "Wed, 21 Oct 2015 07:28:00 GMT"
HUB_NAME = "backend_service.dataset_sources.hugging_face"
Factory = Callable[[dict[str, str] | None], SecureTransport]
Authorizations = list[dict[str, str] | None]


def archive_spec(**overrides: Any) -> HttpsArchiveSourceSpec:
    """Build a validated archive spec that points at the loopback host."""
    fields: dict[str, Any] = {"url": ARCHIVE_URL, "terms_reference": TERMS}
    return HttpsArchiveSourceSpec(**{**fields, **overrides})


def factory_for(server: LoopbackServer) -> tuple[Factory, Authorizations]:
    """Reach the loopback server from a port-443 address the spec accepts.

    The spec only allows port 443, so the connector redirects the TCP hop to
    the server's random port while every address check still runs.
    """
    calls: Authorizations = []

    def connector(address: str, host: str, port: int, timeout: float) -> socket.socket:
        """Connect to the loopback server whatever port the address named."""
        return socket.create_connection((address, server.port), timeout=timeout)

    def factory(authorization: dict[str, str] | None) -> SecureTransport:
        """Record the authorization map and build a short-timeout transport."""
        calls.append(authorization)
        return SecureTransport(
            test_policy,
            authorization=authorization,
            connector=connector,
            read_timeout=5.0,
        )

    return factory, calls


def resolve_against(
    routes: dict[tuple[str, str], Response], spec: HttpsArchiveSourceSpec
) -> tuple[ResolvedSource, list[RecordedRequest], Authorizations]:
    """Resolve the spec against scripted routes and return what the server saw."""
    with LoopbackServer(routes) as server:
        factory, calls = factory_for(server)
        resolved = resolve(spec, transport_factory=factory, uploads=None)
    return resolved, list(server.requests), calls


def describe(
    adapter: SourceAdapter, spec: DatasetSourceSpec, factory: Factory
) -> ResolvedSource:
    """Call an adapter exactly as fetch code does, through the protocol."""
    return adapter.resolve(spec, transport_factory=factory, uploads=None)


def fake_adapter(name: str) -> ModuleType:
    """Build a stand-in adapter module that exposes a resolve function."""
    module = ModuleType(name)

    def resolve(spec: DatasetSourceSpec, **options: object) -> ResolvedSource:
        """Never run; the registry tests only check which module is chosen."""
        raise AssertionError("The stand-in adapter must not resolve anything.")

    vars(module)["resolve"] = resolve
    return module


def test_head_with_size_etag_and_ranges_supports_pause() -> None:
    """A HEAD answer with Content-Length, ETag and byte ranges plans one asset."""
    routes = {
        ("HEAD", f"{ARCHIVE_PATH}?download=1"): Response(
            headers={"Accept-Ranges": "bytes", "Last-Modified": LAST_MODIFIED},
            body=b"z" * 10,
            etag='"abc123"',
        )
    }
    spec = archive_spec(
        url=f"{ARCHIVE_URL}?download=1",
        archive_splits={"DIV2K_valid_HR": "held_out"},
        expected_sha256="a" * 64,
    )
    resolved, requests, calls = resolve_against(routes, spec)
    assert [request.method for request in requests] == ["HEAD"]
    assert "authorization" not in requests[0].headers
    assert calls == [None]
    assert resolved.content == "images"
    assert resolved.access == "available"
    assert resolved.access_guidance is None
    assert resolved.supports_pause is True
    assert resolved.resolved_revision == '"abc123"'
    assert resolved.reference == ARCHIVE_URL
    assert resolved.terms_reference == TERMS
    assert resolved.assets == (
        PlannedAsset(
            url=spec.url,
            path="DIV2K_valid_HR.zip",
            expected_size=10,
            expected_sha256="a" * 64,
            etag='"abc123"',
            resumable=True,
            authorization_host=None,
        ),
    )
    assert resolved.declared_splits is True
    assert resolved.split_mapping == {"div2k_valid_hr": "held_out"}
    assert resolved.member_split_labels == {"DIV2K_valid_HR": "div2k_valid_hr"}
    assert resolved.warnings == ()


@pytest.mark.parametrize("status", [405, 501])
def test_head_refusal_falls_back_to_get_closed_after_headers(status: int) -> None:
    """When HEAD is refused the adapter reads GET headers and drops the body."""
    routes = {
        ("HEAD", ARCHIVE_PATH): Response(status=status),
        ("GET", ARCHIVE_PATH): Response(
            headers={"Accept-Ranges": "bytes"},
            etag='"e1"',
            slow_chunks=[(b"a" * 4, 0.0), (b"b" * 4, 3.0)],
        ),
    }
    started = time.monotonic()
    resolved, requests, _ = resolve_against(routes, archive_spec())
    assert time.monotonic() - started < 2.0
    assert [request.method for request in requests] == ["HEAD", "GET"]
    assert resolved.supports_pause is True
    assert resolved.assets[0].expected_size == 8


@pytest.mark.parametrize(
    ("response", "revision", "size", "warnings"),
    [
        (
            Response(
                headers={"Accept-Ranges": "bytes", "Last-Modified": LAST_MODIFIED},
                body=b"z" * 5,
            ),
            f"{LAST_MODIFIED}+5",
            5,
            (https_archive.RESUME_WARNING,),
        ),
        (
            Response(headers={"Accept-Ranges": "bytes"}, etag='W/"weak"', chunked=True),
            "unversioned",
            None,
            (https_archive.RESUME_WARNING, https_archive.SIZE_WARNING),
        ),
        (
            Response(etag='"strong"', body=b"x"),
            '"strong"',
            1,
            (https_archive.RESUME_WARNING,),
        ),
    ],
)
def test_without_pause_support_the_promise_is_reduced_honestly(
    response: Response, revision: str, size: int | None, warnings: tuple[str, ...]
) -> None:
    """No ETag, a weak ETag or no byte ranges means no resume and a fallback."""
    resolved, _, _ = resolve_against({("HEAD", ARCHIVE_PATH): response}, archive_spec())
    assert resolved.supports_pause is False
    assert resolved.resolved_revision == revision
    assert resolved.assets[0].expected_size == size
    assert resolved.assets[0].resumable is False
    assert resolved.warnings == warnings


@pytest.mark.parametrize(
    ("status", "access"),
    [(404, "not_found"), (401, "access_required"), (403, "access_required")],
)
def test_missing_or_refused_archives_report_access_without_token_guidance(
    status: int, access: str
) -> None:
    """Refusals name the access state plainly and never adopt error page headers."""
    routes = {
        ("HEAD", ARCHIVE_PATH): Response(status=status, body=b"page", etag='"page"')
    }
    resolved, requests, _ = resolve_against(routes, archive_spec())
    assert [request.method for request in requests] == ["HEAD"]
    assert resolved.access == access
    assert resolved.access_guidance is not None
    assert "token" not in resolved.access_guidance.lower()
    assert "hf_" not in resolved.access_guidance.lower()
    assert resolved.resolved_revision == "unversioned"
    assert resolved.supports_pause is False
    assert resolved.assets[0].expected_size is None
    assert resolved.assets[0].etag is None
    assert resolved.warnings == ()


def test_basename_comes_from_the_path_or_falls_back() -> None:
    """Odd address paths still give a safe file name for the cache."""
    assert archive_basename("https://h/a/b/DIV2K%20valid.zip") == "DIV2K valid.zip"
    assert archive_basename("https://h/download/?id=7") == "archive.zip"
    assert archive_basename("https://h/a%2Fb.zip") == "archive.zip"
    assert archive_basename("https://h/.hidden.zip") == "archive.zip"
    assert archive_basename("https://h/a:b.zip") == "archive.zip"
    assert archive_basename("https://h/" + "a" * 256 + ".zip") == "archive.zip"


@pytest.mark.parametrize(
    ("response", "overrides", "code", "status_code"),
    [
        (Response(status=500), {}, "source_unavailable", 502),
        (
            Response(body=b"z" * 9),
            {"maximum_download_bytes": 4},
            "source_too_large",
            400,
        ),
    ],
)
def test_server_errors_and_oversized_archives_fail_before_any_download(
    response: Response, overrides: dict[str, Any], code: str, status_code: int
) -> None:
    """Server errors and known oversized archives are plain, early failures."""
    with pytest.raises(ApplicationFailure) as caught:
        resolve_against({("HEAD", ARCHIVE_PATH): response}, archive_spec(**overrides))
    assert caught.value.code == code
    assert caught.value.status_code == status_code


def test_adapter_for_returns_the_module_for_each_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each source kind maps to the module whose resolve the protocol calls."""
    routes = {("HEAD", ARCHIVE_PATH): Response(body=b"z" * 3, etag='"v1"')}
    spec = archive_spec()
    assert adapter_for(spec).resolve is https_archive.resolve
    with LoopbackServer(routes) as server:
        factory, _ = factory_for(server)
        resolved = describe(adapter_for(spec), spec, factory)
    assert resolved.access == "available"
    hub_module = fake_adapter(HUB_NAME)
    upload_module = fake_adapter("backend_service.dataset_sources.upload_source")
    monkeypatch.setitem(sys.modules, hub_module.__name__, hub_module)
    monkeypatch.setitem(sys.modules, upload_module.__name__, upload_module)
    hub_spec = HuggingFaceSourceSpec(repository="org/name", terms_reference=TERMS)
    upload_spec = UploadSourceSpec(upload_identifier="upload_" + "0" * 32)
    assert adapter_for(hub_spec).resolve is hub_module.resolve
    assert adapter_for(upload_spec).resolve is upload_module.resolve


def test_adapter_for_refuses_missing_or_incomplete_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown kinds and broken adapter modules fail with plain messages."""
    hub_spec = HuggingFaceSourceSpec(repository="org/name", terms_reference=TERMS)
    monkeypatch.setitem(sys.modules, HUB_NAME, ModuleType(HUB_NAME))
    with pytest.raises(ApplicationFailure) as caught:
        adapter_for(hub_spec)
    assert caught.value.code == "source_adapter_unavailable"
    assert caught.value.status_code == 501
    monkeypatch.setitem(sys.modules, HUB_NAME, None)
    with pytest.raises(ApplicationFailure) as caught:
        adapter_for(hub_spec)
    assert caught.value.code == "source_adapter_unavailable"
    monkeypatch.delitem(registry.ADAPTER_MODULES, "https_archive")
    with pytest.raises(ApplicationFailure) as caught:
        adapter_for(archive_spec())
    assert caught.value.code == "source_kind_unsupported"
    assert caught.value.status_code == 422
