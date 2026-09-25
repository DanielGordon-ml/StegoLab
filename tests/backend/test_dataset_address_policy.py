"""Address validation and resolved-address checks for dataset downloads."""

import ipaddress
import socket

import pytest
from dataset_source_fixtures import resolver_answering, test_policy

from backend_service.dataset_sources.address_policy import (
    AddressPolicy,
    check_address,
    resolve_and_check,
    validate_url,
)
from backend_service.failures import ApplicationFailure

PRODUCTION = AddressPolicy()
PUBLIC_IPV4 = "93.184.216.34"
PUBLIC_IPV6 = "2606:2800:220:1:248:1893:25c8:1946"
BLOCKED_ADDRESSES = [
    "127.0.0.1",
    "10.0.0.1",
    "172.16.5.5",
    "192.168.1.1",
    "169.254.169.254",
    "100.64.1.1",
    "224.0.0.1",
    "0.0.0.0",
    "::1",
    "fe80::1",
    "fc00::1",
    "::ffff:10.0.0.1",
    "2002:0a00:0001::1",
    "2001:0:c000:201:0:0:f5ff:fffe",
]
REFUSED_URLS = [
    "http://example.com/file",
    "https://user@example.com/file",
    "https://user:secret@example.com/file",
    "https://example.com/file#part",
    "https://example.com:8443/file",
    "https://example.com:abc/file",
    "https:///file",
    "https://:443/file",
    "https://exam ple.com/file",
    "https://exämple.com/file",
    "https://example.com/" + "a" * 4096,
    "https://[not-an-address]/file",
    "ftp://example.com/file",
]


@pytest.mark.parametrize("url", REFUSED_URLS)
def test_refused_urls_share_one_message(url: str) -> None:
    """Every malformed or non-https address gets the same fixed explanation."""
    with pytest.raises(ApplicationFailure) as failure:
        validate_url(url)
    assert failure.value.code == "download_address"
    assert failure.value.status_code == 400
    assert "example" not in failure.value.message


def test_accepted_url_parts() -> None:
    """A plain https address splits into host, port and request target."""
    checked = validate_url("https://Example.com/data/file.zip?download=1")
    assert checked.host == "example.com"
    assert checked.port == 443
    assert checked.path_and_query == "/data/file.zip?download=1"
    assert validate_url("https://example.com").path_and_query == "/"
    assert validate_url("https://example.com:443/x").port == 443
    assert validate_url(f"https://[{PUBLIC_IPV6}]/x").host == PUBLIC_IPV6


def test_policy_ports_extend_the_default() -> None:
    """A test policy may add its server port; production keeps 443 alone."""
    policy = AddressPolicy(allow_loopback=True, allowed_ports=frozenset({443, 8443}))
    assert validate_url("https://127.0.0.1:8443/file", policy).port == 8443
    assert validate_url("https://127.0.0.1/file", policy).port == 443
    with pytest.raises(ApplicationFailure):
        validate_url("https://127.0.0.1:8443/file", PRODUCTION)


@pytest.mark.parametrize("address", BLOCKED_ADDRESSES)
def test_blocked_addresses(address: str) -> None:
    """Private, local, reserved and embedded-private addresses are refused."""
    with pytest.raises(ApplicationFailure) as failure:
        check_address(ipaddress.ip_address(address), PRODUCTION)
    assert failure.value.code == "download_address_blocked"
    assert address not in failure.value.message


@pytest.mark.parametrize("address", [PUBLIC_IPV4, PUBLIC_IPV6])
def test_public_addresses_pass(address: str) -> None:
    """Global unicast addresses pass under the production policy."""
    check_address(ipaddress.ip_address(address), PRODUCTION)


def test_loopback_only_with_the_test_policy() -> None:
    """Loopback passes only when allowed; private ranges stay blocked."""
    for address in ("127.0.0.1", "::1"):
        check_address(ipaddress.ip_address(address), test_policy)
        with pytest.raises(ApplicationFailure):
            check_address(ipaddress.ip_address(address), PRODUCTION)
    with pytest.raises(ApplicationFailure):
        check_address(ipaddress.ip_address("10.0.0.1"), test_policy)


def test_address_literals_skip_name_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """An address literal is checked directly and never sent to the resolver."""
    monkeypatch.setattr(socket, "getaddrinfo", _refuse_lookup)
    assert resolve_and_check(PUBLIC_IPV4, PRODUCTION) == [PUBLIC_IPV4]
    assert resolve_and_check("127.0.0.1", test_policy) == ["127.0.0.1"]
    with pytest.raises(ApplicationFailure) as failure:
        resolve_and_check("169.254.169.254", PRODUCTION)
    assert failure.value.code == "download_address_blocked"


def test_one_private_answer_blocks_the_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host that also resolves to a private address is refused entirely."""
    answers = {"mirror.example": [PUBLIC_IPV4, "10.0.0.1"]}
    monkeypatch.setattr(socket, "getaddrinfo", resolver_answering(answers))
    with pytest.raises(ApplicationFailure) as failure:
        resolve_and_check("mirror.example", PRODUCTION)
    assert failure.value.code == "download_address_blocked"
    assert "mirror" not in failure.value.message


def test_public_answers_keep_resolver_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """All-public answers come back in resolver order without duplicates."""
    answers = {"mirror.example": [PUBLIC_IPV6, PUBLIC_IPV4, PUBLIC_IPV4]}
    monkeypatch.setattr(socket, "getaddrinfo", resolver_answering(answers))
    assert resolve_and_check("mirror.example", PRODUCTION) == [
        PUBLIC_IPV6,
        PUBLIC_IPV4,
    ]


def test_unresolved_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """A resolver failure becomes the fixed unresolved-host message."""
    monkeypatch.setattr(socket, "getaddrinfo", _fail_lookup)
    with pytest.raises(ApplicationFailure) as failure:
        resolve_and_check("missing.example", PRODUCTION)
    assert failure.value.code == "download_address_unresolved"
    assert "missing" not in failure.value.message


def _refuse_lookup(*arguments: object, **options: object) -> None:
    """Fail the test if the resolver is consulted at all."""
    pytest.fail("The resolver must not be used for address literals.")


def _fail_lookup(*arguments: object, **options: object) -> None:
    """Behave like a resolver that cannot find the name."""
    raise socket.gaierror(8, "name not known")
