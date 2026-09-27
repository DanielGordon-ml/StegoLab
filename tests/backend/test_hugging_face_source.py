"""The Hugging Face adapter: revision, listing, selection, access and the token."""

from collections.abc import Callable
from typing import Any

import hub_fixtures as fixtures
import pytest
from hub_fixtures import (
    PUBLIC_PARQUET_FILES,
    PUBLIC_REPOSITORY,
    PUBLIC_SHA,
    FakeHub,
    Reply,
    RouteTable,
    json_reply,
    public_parquet_routes,
    revision_url,
    tree_url,
)

from backend_service.dataset_sources.hugging_face import (
    GATED_GUIDANCE,
    GATED_WARNING,
    MAXIMUM_DOCUMENT_BYTES,
    MAXIMUM_TREE_PAGES,
    resolve,
)
from backend_service.dataset_sources.plans import ResolvedSource
from backend_service.failures import ApplicationFailure
from schemas.dataset_sources import HuggingFaceSourceSpec

FAKE_TOKEN = "hf_fake_token_for_tests_only"
HUB = "https://huggingface.co/datasets"
VALIDATION_SHARD = "data/validation-00000-of-00001.parquet"
NAMED_FILES = ["notes/extra.txt", "data/test-00000-of-00001.parquet"]


@pytest.fixture(autouse=True)
def no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test without a server token, whatever the shell exported."""
    for variable in ("HF_TOKEN", "HF_ACCESS_TOKEN", "HF_TOKEN_FILE"):
        monkeypatch.delenv(variable, raising=False)


def hub_spec(
    repository: str = PUBLIC_REPOSITORY, **overrides: Any
) -> HuggingFaceSourceSpec:
    """Build a Hugging Face spec with optional field overrides."""
    fields: dict[str, Any] = {"repository": repository, "terms_reference": HUB}
    return HuggingFaceSourceSpec(**{**fields, **overrides})


def resolve_with(
    hub: FakeHub, spec: HuggingFaceSourceSpec, sleeps: list[float] | None = None
) -> ResolvedSource:
    """Resolve through the fake hub, collecting any retry sleeps."""
    recorded = [] if sleeps is None else sleeps
    factory = hub.transport_factory
    return resolve(spec, transport_factory=factory, uploads=None, sleep=recorded.append)


def paths(resolved: ResolvedSource) -> list[str]:
    """List the asset paths in the order the adapter returned them."""
    return [asset.path for asset in resolved.assets]


def test_public_dataset_resolves_sha_follows_pages_and_lists_assets() -> None:
    """The revision becomes a sha, both pages are read and assets are complete."""
    hub = FakeHub(public_parquet_routes())
    resolved = resolve_with(hub, hub_spec())
    assert resolved.access == "available" and resolved.access_guidance is None
    assert resolved.requested_revision == "main"
    assert resolved.resolved_revision == PUBLIC_SHA
    assert resolved.reference == f"{HUB}/{PUBLIC_REPOSITORY}/tree/{PUBLIC_SHA}"
    assert [request.url for request in hub.requests] == [
        revision_url(PUBLIC_REPOSITORY),
        tree_url(PUBLIC_REPOSITORY, PUBLIC_SHA),
        tree_url(PUBLIC_REPOSITORY, PUBLIC_SHA, cursor=fixtures.SECOND_PAGE_CURSOR),
    ]
    assert paths(resolved) == [*PUBLIC_PARQUET_FILES, "notes/extra.txt"]
    assert resolved.supports_pause and not resolved.declared_splits
    assert resolved.split_mapping == {} and resolved.member_split_labels == {}
    assert resolved.warnings == ("Skipped 1 file with unsafe names.",)
    assert resolved.content == "images" and resolved.terms_reference == HUB
    assert all(response.closed for response in hub.transports[0].responses)
    parquet, text = resolved.assets[0], resolved.assets[-1]
    resolve_base = f"{HUB}/{PUBLIC_REPOSITORY}/resolve/{PUBLIC_SHA}"
    assert parquet.url == f"{resolve_base}/{parquet.path}"
    assert parquet.expected_size == fixtures.PUBLIC_FILE_SIZES[parquet.path]
    assert parquet.expected_sha256 == fixtures.lfs_digest(parquet.path)
    assert parquet.etag is None and parquet.resumable
    assert parquet.authorization_host == "huggingface.co"
    assert text.path == "notes/extra.txt" and text.expected_sha256 is None
    assert text.expected_size == fixtures.PUBLIC_FILE_SIZES["notes/extra.txt"]


def test_split_keeps_matching_shards_and_maps_labels() -> None:
    """A requested split keeps its shards and labels them for the materializer."""
    resolved = resolve_with(
        FakeHub(public_parquet_routes()), hub_spec(split="validation")
    )
    assert paths(resolved) == [VALIDATION_SHARD]
    assert resolved.split_mapping == {"validation": "tuning"}
    basename = VALIDATION_SHARD.removeprefix("data/")
    assert resolved.member_split_labels == {basename: "validation"}
    assert resolved.declared_splits
    spec = hub_spec(split="validation", archive_splits={"validation": "held_out"})
    overridden = resolve_with(FakeHub(public_parquet_routes()), spec)
    assert overridden.split_mapping == {"validation": "held_out"}


@pytest.mark.parametrize(
    ("overrides", "listed_prefix", "expected"),
    [
        ({"path_prefix": "data"}, "data", list(PUBLIC_PARQUET_FILES)),
        (
            {"path_prefix": "data", "split": "train", "file_names": NAMED_FILES},
            None,
            sorted(NAMED_FILES),
        ),
    ],
)
def test_prefix_scopes_the_listing_but_named_files_beat_every_filter(
    overrides: dict[str, Any], listed_prefix: str | None, expected: list[str]
) -> None:
    """A prefix narrows the listing address; named files search the whole tree."""
    hub = FakeHub(public_parquet_routes())
    resolved = resolve_with(hub, hub_spec(**overrides))
    listing = tree_url(PUBLIC_REPOSITORY, PUBLIC_SHA, prefix=listed_prefix)
    assert hub.requests[1].url == listing and paths(resolved) == expected


def test_maximum_files_keeps_the_first_paths_with_a_warning() -> None:
    """More matches than allowed keep the first ones in path order and warn."""
    resolved = resolve_with(FakeHub(public_parquet_routes()), hub_spec(maximum_files=2))
    assert paths(resolved) == list(PUBLIC_PARQUET_FILES[:2])
    assert any(note.startswith("Only the first 2 of 5") for note in resolved.warnings)


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"file_names": ["data/absent.parquet"]}, "source_files_missing"),
        ({"file_names": ["README.md"]}, "source_files_unsupported"),
        ({"path_prefix": "absent"}, "source_files_empty"),
        (
            {"repository": fixtures.IMAGE_REPOSITORY, "split": "train"},
            "source_files_empty",
        ),
    ],
)
def test_selection_problems_fail_plainly(overrides: dict[str, Any], code: str) -> None:
    """Missing, unsupported or unmatched selections raise fixed failure codes."""
    routes = {**public_parquet_routes(), **fixtures.image_folder_routes()}
    with pytest.raises(ApplicationFailure) as raised:
        resolve_with(FakeHub(routes), hub_spec(**overrides))
    assert raised.value.code == code


def test_image_folder_keeps_supported_files_only() -> None:
    """Png, txt and zip files are kept; readme and binary files are not."""
    hub = FakeHub(fixtures.image_folder_routes())
    resolved = resolve_with(hub, hub_spec(fixtures.IMAGE_REPOSITORY))
    assert paths(resolved) == list(fixtures.IMAGE_FILES)
    assert resolved.resolved_revision == fixtures.IMAGE_SHA and resolved.warnings == ()
    assert resolved.assets[1].expected_sha256 == fixtures.lfs_digest("images/cat.png")
    assert resolved.assets[3].expected_sha256 is None


@pytest.mark.parametrize(
    ("routes", "repository", "revision", "request_count"),
    [
        (fixtures.gated_routes(401), fixtures.GATED_REPOSITORY, "unresolved", 1),
        (fixtures.gated_routes(403), fixtures.GATED_REPOSITORY, "unresolved", 1),
        (fixtures.gated_tree_routes(), fixtures.GATED_TREE_REPOSITORY, PUBLIC_SHA, 2),
        (fixtures.missing_routes(), fixtures.MISSING_REPOSITORY, "unresolved", 1),
    ],
)
def test_denied_and_missing_answers_stop_before_any_download(
    routes: RouteTable, repository: str, revision: str, request_count: int
) -> None:
    """Denied lookups carry the fixed guidance; an unknown repository carries none."""
    hub = FakeHub(routes)
    resolved = resolve_with(hub, hub_spec(repository))
    missing = repository == fixtures.MISSING_REPOSITORY
    assert resolved.access == ("not_found" if missing else "access_required")
    assert resolved.access_guidance == (None if missing else GATED_GUIDANCE)
    assert resolved.assets == () and resolved.resolved_revision == revision
    assert len(hub.requests) == request_count


def test_gated_flag_needs_a_token_before_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dataset marked gated is not listed without a token; with one it is."""
    hub = FakeHub(fixtures.gated_flag_routes())
    spec = hub_spec(fixtures.GATED_FLAG_REPOSITORY)
    assert resolve_with(hub, spec).access == "access_required"
    assert len(hub.requests) == 1
    monkeypatch.setenv("HF_ACCESS_TOKEN", FAKE_TOKEN)
    resolved = resolve_with(hub, spec)
    assert resolved.access == "available" and len(resolved.assets) == 1
    assert GATED_WARNING in resolved.warnings


def test_rate_limited_answers_are_retried_after_a_short_sleep() -> None:
    """A 429 is retried once after the fixed delay and then succeeds."""
    hub = FakeHub(fixtures.rate_limited_routes())
    sleeps: list[float] = []
    resolved = resolve_with(hub, hub_spec(fixtures.RATE_LIMITED_REPOSITORY), sleeps)
    assert resolved.access == "available" and len(resolved.assets) == 1
    assert sleeps == [0.5]
    revision = revision_url(fixtures.RATE_LIMITED_REPOSITORY)
    assert [request.url for request in hub.requests[:2]] == [revision, revision]


def test_server_errors_give_up_after_the_retry_limit() -> None:
    """Repeated server errors end in one plain failure after three retries."""
    hub = FakeHub(fixtures.broken_routes())
    sleeps: list[float] = []
    with pytest.raises(ApplicationFailure) as raised:
        resolve_with(hub, hub_spec(fixtures.BROKEN_REPOSITORY), sleeps)
    assert raised.value.code == "source_unavailable"
    assert raised.value.status_code == 502
    assert sleeps == [0.5] * 3 and len(hub.requests) == 4


@pytest.mark.parametrize("token", [None, FAKE_TOKEN])
def test_token_is_passed_only_as_a_hub_host_map(
    monkeypatch: pytest.MonkeyPatch, token: str | None
) -> None:
    """With a token the factory gets one bearer entry for the Hub; else None."""
    if token is not None:
        monkeypatch.setenv("HF_ACCESS_TOKEN", token)
    hub = FakeHub(public_parquet_routes())
    resolve_with(hub, hub_spec())
    expected = None if token is None else f"Bearer {token}"
    assert [transport.authorization for transport in hub.transports] == [
        None if expected is None else {"huggingface.co": expected}
    ]
    sent = [request.headers.get("Authorization") for request in hub.requests]
    assert sent == [expected] * 3


def test_token_never_appears_in_records_messages_or_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The token stays inside the transport; nothing returned or raised has it."""
    monkeypatch.setenv("HF_ACCESS_TOKEN", FAKE_TOKEN)
    caplog.set_level("DEBUG")
    resolved = resolve_with(FakeHub(public_parquet_routes()), hub_spec())
    gated_spec = hub_spec(fixtures.GATED_REPOSITORY)
    gated = resolve_with(FakeHub(fixtures.gated_routes()), gated_spec)
    with pytest.raises(ApplicationFailure) as raised:
        broken_spec = hub_spec(fixtures.BROKEN_REPOSITORY)
        resolve_with(FakeHub(fixtures.broken_routes()), broken_spec)
    assert FAKE_TOKEN not in repr(resolved) and FAKE_TOKEN not in repr(gated)
    assert (
        FAKE_TOKEN not in str(raised.value) and FAKE_TOKEN not in raised.value.message
    )
    assert FAKE_TOKEN not in caplog.text


def test_listing_stops_after_the_page_limit() -> None:
    """A listing that keeps pointing at another page fails after fifty pages."""
    routes = public_parquet_routes()
    for page in range(MAXIMUM_TREE_PAGES + 1):
        cursor = None if page == 0 else str(page)
        following = tree_url(PUBLIC_REPOSITORY, PUBLIC_SHA, cursor=str(page + 1))
        entries = [fixtures.file_entry(f"pages/{page}.txt", 1)]
        key = ("GET", tree_url(PUBLIC_REPOSITORY, PUBLIC_SHA, cursor=cursor))
        routes[key] = json_reply(200, entries, **fixtures.link_header(following))
    with pytest.raises(ApplicationFailure) as raised:
        resolve_with(FakeHub(routes), hub_spec())
    assert raised.value.code == "source_listing_limit"


@pytest.mark.parametrize(
    ("key_for", "reply"),
    [
        (revision_url, json_reply(200, {"sha": "not-a-sha"})),
        (revision_url, json_reply(200, [1])),
        (revision_url, (200, {}, b"<html>not json</html>")),
        (revision_url, (200, {}, bytes(MAXIMUM_DOCUMENT_BYTES + 1))),
        (
            lambda repository: tree_url(repository, PUBLIC_SHA),
            json_reply(200, [], **fixtures.link_header("https://example.test/tree")),
        ),
    ],
)
def test_damaged_answers_fail_plainly(
    key_for: Callable[[str], str], reply: Reply
) -> None:
    """Bad shas, wrong shapes, non-JSON, oversized bodies and off-host links."""
    routes = public_parquet_routes()
    routes[("GET", key_for(PUBLIC_REPOSITORY))] = reply
    with pytest.raises(ApplicationFailure) as raised:
        resolve_with(FakeHub(routes), hub_spec())
    assert raised.value.code == "source_listing_invalid"
