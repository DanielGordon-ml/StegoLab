"""Address rules that keep dataset downloads on public https hosts."""

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit

from backend_service.failures import ApplicationFailure

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
MAXIMUM_URL_LENGTH = 4096
HOST_CHARACTERS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.-"
)
SHARED_ADDRESS_SPACE = ipaddress.IPv4Network("100.64.0.0/10")
_MESSAGES = {
    "download_address": (
        "The download address must use https on port 443 without credentials "
        "or fragments."
    ),
    "download_address_blocked": (
        "The download address points to a private, local or reserved network "
        "and was refused."
    ),
    "download_address_unresolved": "The download host name could not be resolved.",
}


def address_failure(code: str) -> ApplicationFailure:
    """Return the fixed message for an address failure code."""
    return ApplicationFailure(code, _MESSAGES[code], 400)


@dataclass(frozen=True)
class AddressPolicy:
    """Decide which resolved addresses and ports a download may use.

    The defaults are the production policy: loopback is blocked and only port
    443 is accepted. Tests pass ``allow_loopback=True`` and add the random port
    of their loopback server to ``allowed_ports``; production keeps {443}.
    """

    allow_loopback: bool = False
    allowed_ports: frozenset[int] = frozenset({443})


PRODUCTION_POLICY = AddressPolicy()


@dataclass(frozen=True)
class ValidatedUrl:
    """A checked https address split into the parts a connection needs."""

    url: str
    host: str
    port: int
    path_and_query: str


def validate_url(url: str, policy: AddressPolicy = PRODUCTION_POLICY) -> ValidatedUrl:
    """Accept a plain ASCII https address on an allowed port, or refuse it."""
    if len(url) > MAXIMUM_URL_LENGTH or "#" in url:
        raise address_failure("download_address")
    if any(character < "!" or character > "~" for character in url):
        raise address_failure("download_address")
    try:
        parts = urlsplit(url)
        explicit_port = parts.port
    except ValueError:
        raise address_failure("download_address") from None
    host = parts.hostname
    if parts.scheme != "https" or not host or "@" in parts.netloc:
        raise address_failure("download_address")
    if parts.netloc.startswith("["):
        _parse_literal(host)
    elif not set(host) <= HOST_CHARACTERS:
        raise address_failure("download_address")
    port = 443 if explicit_port is None else explicit_port
    if port not in policy.allowed_ports:
        raise address_failure("download_address")
    path_and_query = parts.path or "/"
    if parts.query:
        path_and_query += "?" + parts.query
    return ValidatedUrl(url=url, host=host, port=port, path_and_query=path_and_query)


def check_address(address: IPAddress, policy: AddressPolicy) -> None:
    """Refuse any address (or embedded IPv4 address) that is not public."""
    candidate = _embedded_ipv4(address)
    if candidate.is_loopback:
        if policy.allow_loopback:
            return
        raise address_failure("download_address_blocked")
    blocked = (
        candidate.is_multicast
        or candidate.is_reserved
        or candidate.is_unspecified
        or candidate.is_link_local
        or candidate.is_private
    )
    if isinstance(candidate, ipaddress.IPv4Address):
        blocked = blocked or candidate in SHARED_ADDRESS_SPACE
    if blocked or not candidate.is_global:
        raise address_failure("download_address_blocked")


def resolve_and_check(host: str, policy: AddressPolicy) -> list[str]:
    """Resolve a host once; every answer must pass, and the first one is used."""
    try:
        literal: IPAddress | None = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        check_address(literal, policy)
        return [str(literal)]
    try:
        records = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError):
        raise address_failure("download_address_unresolved") from None
    addresses: list[str] = []
    for record in records:
        text = str(record[4][0])
        check_address(_parse_literal(text, "download_address_blocked"), policy)
        if text not in addresses:
            addresses.append(text)
    if not addresses:
        raise address_failure("download_address_unresolved")
    return addresses


def _parse_literal(text: str, code: str = "download_address") -> IPAddress:
    """Parse an address literal or raise the given failure code."""
    try:
        return ipaddress.ip_address(text)
    except ValueError:
        raise address_failure(code) from None


def _embedded_ipv4(address: IPAddress) -> IPAddress:
    """Return the IPv4 address carried by a transition address, else the input."""
    if isinstance(address, ipaddress.IPv4Address):
        return address
    if address.ipv4_mapped is not None:
        return address.ipv4_mapped
    if address.sixtofour is not None:
        return address.sixtofour
    return address if address.teredo is None else address.teredo[1]
