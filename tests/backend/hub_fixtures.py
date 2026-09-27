"""Scripted Hugging Face Hub replies and a transport stand-in for adapter tests."""

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal
from urllib.parse import quote, urlsplit

Reply = tuple[int, dict[str, str], bytes]
Route = tuple[tuple[str, str], Reply]
RouteTable = dict[tuple[str, str], Reply | list[Reply]]

API_BASE = "https://huggingface.co/api/datasets"
PUBLIC_REPOSITORY = "org/public-parquet"
PUBLIC_SHA = "0123456789abcdef0123456789abcdef01234567"
IMAGE_REPOSITORY = "org/image-folder"
IMAGE_SHA = "89abcdef" * 5
GATED_REPOSITORY = "org/gated-data"
GATED_TREE_REPOSITORY = "org/gated-tree"
GATED_FLAG_REPOSITORY = "org/gated-flag"
MISSING_REPOSITORY = "org/missing-data"
RATE_LIMITED_REPOSITORY = "org/rate-limited"
BROKEN_REPOSITORY = "org/broken-hub"
SECOND_PAGE_CURSOR = "page-two"
SHARD = "data/train-00000-of-00001.parquet"
PUBLIC_FILE_SIZES = {
    "data/test-00000-of-00001.parquet": 950_000,
    "data/train-00000-of-00002.parquet": 5_000_000,
    "data/train-00001-of-00002.parquet": 4_800_000,
    "data/validation-00000-of-00001.parquet": 900_000,
    "notes/extra.txt": 640,
}
PUBLIC_PARQUET_FILES = tuple(
    path for path in sorted(PUBLIC_FILE_SIZES) if path.endswith(".parquet")
)
IMAGE_FILES = ("bundle.zip", "images/cat.png", "images/dog.png", "images/notes.txt")
DENIED = {"error": "Access to this dataset is restricted. Please log in."}


def lfs_digest(path: str) -> str:
    """Derive a stable sha256 for a fixture file from its path."""
    return hashlib.sha256(path.encode()).hexdigest()


def file_entry(path: str, size: int, *, lfs: bool = False) -> dict[str, object]:
    """Describe one file the way the Hub tree listing does."""
    entry: dict[str, object] = {"type": "file", "size": size, "path": path}
    if lfs:
        entry["lfs"] = {"oid": lfs_digest(path), "size": size, "pointerSize": 134}
    return entry


def directory_entry(path: str) -> dict[str, object]:
    """Describe one folder the way the Hub tree listing does."""
    return {"type": "directory", "oid": "c" * 40, "size": 0, "path": path}


def json_reply(status: int, document: object, **headers: str) -> Reply:
    """Build one scripted JSON reply with optional extra headers."""
    headers = {"content-type": "application/json", **headers}
    return status, headers, json.dumps(document).encode()


def revision_url(repository: str, revision: str = "main") -> str:
    """Build the revision address the adapter is expected to request."""
    return f"{API_BASE}/{repository}/revision/{quote(revision, safe='')}"


def tree_url(
    repository: str, sha: str, prefix: str | None = None, cursor: str | None = None
) -> str:
    """Build the recursive listing address, with an optional prefix and cursor."""
    scope = "" if prefix is None else f"/{prefix}"
    query = "?recursive=true" if cursor is None else f"?recursive=true&cursor={cursor}"
    return f"{API_BASE}/{repository}/tree/{sha}{scope}{query}"


def link_header(url: str) -> dict[str, str]:
    """Build the pagination header that names the next listing page."""
    return {"link": f'<{url}>; rel="next"'}


def routes(*scripted: Route) -> RouteTable:
    """Collect scripted routes into a table the fake transport answers from."""
    return {key: reply for key, reply in scripted}


def revision_route(repository: str, sha: str, *, gated: bool | str = False) -> Route:
    """Script a successful revision lookup the way the Hub answers it."""
    siblings = [{"rfilename": "README.md"}]
    document = {"id": repository, "sha": sha, "gated": gated, "siblings": siblings}
    return ("GET", revision_url(repository)), json_reply(200, document)


def tree_route(
    repository: str,
    sha: str,
    entries: list[dict[str, object]],
    *,
    prefix: str | None = None,
    cursor: str | None = None,
    next_cursor: str | None = None,
) -> Route:
    """Script one listing page, optionally pointing at a following page."""
    following = tree_url(repository, sha, cursor=next_cursor)
    headers = {} if next_cursor is None else link_header(following)
    key = ("GET", tree_url(repository, sha, prefix, cursor))
    return key, json_reply(200, entries, **headers)


def public_parquet_routes() -> RouteTable:
    """Script a public parquet dataset listed over two pages plus a prefix page."""
    sizes = PUBLIC_FILE_SIZES
    shards = [file_entry(path, sizes[path], lfs=True) for path in PUBLIC_PARQUET_FILES]
    test_shard, train_one, train_two, validation_shard = shards
    first_page = [
        file_entry(".gitattributes", 2500),
        file_entry("README.md", 1200),
        directory_entry("data"),
        train_one,
        train_two,
    ]
    second_page = [
        validation_shard,
        test_shard,
        directory_entry("notes"),
        file_entry("notes/extra.txt", sizes["notes/extra.txt"]),
        file_entry("notes/odd:name.txt", 10),
    ]
    repository, sha, cursor = PUBLIC_REPOSITORY, PUBLIC_SHA, SECOND_PAGE_CURSOR
    return routes(
        revision_route(repository, sha),
        tree_route(repository, sha, first_page, next_cursor=cursor),
        tree_route(repository, sha, second_page, cursor=cursor),
        tree_route(repository, sha, shards, prefix="data"),
    )


def image_folder_routes() -> RouteTable:
    """Script an image-folder dataset with png files, a text file and a zip."""
    page = [
        file_entry("README.md", 300),
        directory_entry("images"),
        file_entry("images/cat.png", 4096, lfs=True),
        file_entry("images/dog.png", 5120, lfs=True),
        file_entry("images/notes.txt", 12),
        file_entry("images/raw.bin", 99, lfs=True),
        file_entry("bundle.zip", 70_000, lfs=True),
    ]
    return routes(
        revision_route(IMAGE_REPOSITORY, IMAGE_SHA),
        tree_route(IMAGE_REPOSITORY, IMAGE_SHA, page),
    )


def gated_routes(status: int = 403) -> RouteTable:
    """Script a gated dataset whose revision lookup is denied outright."""
    return {("GET", revision_url(GATED_REPOSITORY)): json_reply(status, DENIED)}


def gated_tree_routes() -> RouteTable:
    """Script a dataset whose revision resolves but whose listing is denied."""
    listing = ("GET", tree_url(GATED_TREE_REPOSITORY, PUBLIC_SHA))
    return routes(
        revision_route(GATED_TREE_REPOSITORY, PUBLIC_SHA),
        (listing, json_reply(403, DENIED)),
    )


def gated_flag_routes() -> RouteTable:
    """Script a dataset the Hub marks as gated while still listing its files."""
    shard = file_entry(SHARD, 100, lfs=True)
    return routes(
        revision_route(GATED_FLAG_REPOSITORY, PUBLIC_SHA, gated="auto"),
        tree_route(GATED_FLAG_REPOSITORY, PUBLIC_SHA, [shard]),
    )


def missing_routes() -> RouteTable:
    """Script a repository the Hub does not know."""
    reply = json_reply(404, {"error": "Repository not found"})
    return {("GET", revision_url(MISSING_REPOSITORY)): reply}


def rate_limited_routes() -> RouteTable:
    """Script a revision lookup answered 429 once, then 200, then a listing."""
    key, success = revision_route(RATE_LIMITED_REPOSITORY, PUBLIC_SHA)
    limited = json_reply(429, {"error": "Too many requests"}, **{"retry-after": "1"})
    shard = file_entry(SHARD, 100, lfs=True)
    scripted = routes(tree_route(RATE_LIMITED_REPOSITORY, PUBLIC_SHA, [shard]))
    scripted[key] = [limited, success]
    return scripted


def broken_routes() -> RouteTable:
    """Script a Hub that keeps answering with a server error."""
    reply = json_reply(503, {"error": "Service unavailable"})
    return {("GET", revision_url(BROKEN_REPOSITORY)): reply}


@dataclass(frozen=True)
class FakeRequest:
    """What one request looked like when it reached the fake transport."""

    method: str
    url: str
    headers: dict[str, str]


class FakeResponse:
    """A scripted reply exposing the same surface the adapter reads."""

    def __init__(self, reply: Reply) -> None:
        """Keep the status, headers and body of one scripted reply."""
        self.status: int = reply[0]
        self.headers: dict[str, str] = dict(reply[1])
        self._body = reply[2]
        self.closed = False

    def iter_bytes(self, chunk_size: int = 65536) -> Iterator[bytes]:
        """Yield the body in pieces of the requested size."""
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start : start + chunk_size]

    def close(self) -> None:
        """Remember that the adapter released the response."""
        self.closed = True


class FakeTransport:
    """Answer requests from a route table and record the headers per address."""

    def __init__(
        self,
        routes: RouteTable,
        authorization: dict[str, str] | None = None,
        log: list[FakeRequest] | None = None,
    ) -> None:
        """Keep the routes, the exact-host authorization map and a request log."""
        self.routes = routes
        self.authorization = None if authorization is None else dict(authorization)
        self.requests: list[FakeRequest] = [] if log is None else log
        self.responses: list[FakeResponse] = []

    def open(
        self,
        method: Literal["GET", "HEAD"],
        url: str,
        *,
        headers: dict[str, str] | None = None,
        byte_range: tuple[int, int | None] | None = None,
        if_range: str | None = None,
    ) -> FakeResponse:
        """Record the request as the real transport would send it, then reply."""
        sent = {
            name: value
            for name, value in (headers or {}).items()
            if name.lower() not in {"authorization", "host"}
        }
        host = urlsplit(url).hostname
        if self.authorization and host in self.authorization:
            sent["Authorization"] = self.authorization[host]
        self.requests.append(FakeRequest(method, url, sent))
        response = FakeResponse(self._reply(method, url))
        self.responses.append(response)
        return response

    def _reply(self, method: str, url: str) -> Reply:
        """Pick the scripted reply; sequences advance until their last entry."""
        scripted = self.routes.get((method, url))
        if scripted is None:
            return json_reply(404, {"error": "Not Found"})
        if isinstance(scripted, list):
            return scripted.pop(0) if len(scripted) > 1 else scripted[0]
        return scripted


class FakeHub:
    """Build transports over one route table and remember every one of them."""

    def __init__(self, routes: RouteTable) -> None:
        """Keep the routes shared by every transport this hub builds."""
        self.routes = routes
        self.transports: list[FakeTransport] = []
        self.requests: list[FakeRequest] = []

    def transport_factory(self, authorization: dict[str, str] | None) -> FakeTransport:
        """Create a transport the way the fetch layer would, keeping a handle."""
        transport = FakeTransport(self.routes, authorization, self.requests)
        self.transports.append(transport)
        return transport
