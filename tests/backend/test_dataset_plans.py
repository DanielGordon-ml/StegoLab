"""Identities, labels, plans and inspections for remote dataset sources."""

import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from backend_service.dataset_sources.cache import DatasetCache
from backend_service.dataset_sources.plans import (
    PlannedAsset,
    ResolvedSource,
    asset_identity,
    inspection_for,
    materialization_identity,
    plan_for,
    suggested_source_name,
    upstream_label,
)
from backend_service.disk_reserve import free_disk_bytes, minimum_free_bytes
from backend_service.failures import ApplicationFailure
from schemas.dataset_sources import (
    DatasetSourceSpec,
    HttpsArchiveSourceSpec,
    HuggingFaceSourceSpec,
    UploadSourceSpec,
)

COUNTER_KEY = re.compile(r"[a-z][a-z0-9_]{0,63}")
SHA = "1" * 40
DIGEST = "d" * 64


def hub_spec(**overrides: Any) -> HuggingFaceSourceSpec:
    """Build a Hugging Face spec with optional field overrides."""
    fields: dict[str, Any] = {
        "repository": "org/My-Data",
        "terms_reference": "https://huggingface.co/datasets/org/My-Data",
    }
    return HuggingFaceSourceSpec(**{**fields, **overrides})


def archive_spec(**overrides: Any) -> HttpsArchiveSourceSpec:
    """Build an https archive spec with optional field overrides."""
    fields: dict[str, Any] = {
        "url": "https://example.test/files/DIV2K_valid_HR.zip",
        "terms_reference": "https://example.test/terms",
    }
    return HttpsArchiveSourceSpec(**{**fields, **overrides})


def planned(path: str, size: int | None = 100) -> PlannedAsset:
    """Describe one remote file of a known or unknown size."""
    return PlannedAsset(
        url=f"https://example.test/files/{path}",
        path=path,
        expected_size=size,
        expected_sha256=None,
        etag=None,
        resumable=True,
        authorization_host=None,
    )


def resolved(spec: DatasetSourceSpec, **overrides: Any) -> ResolvedSource:
    """Resolve a spec the way an adapter would, with optional overrides."""
    if isinstance(spec, HuggingFaceSourceSpec):
        reference = f"https://huggingface.co/datasets/{spec.repository}/tree/{SHA}"
    elif isinstance(spec, UploadSourceSpec):
        reference = "upload:Photos Set.zip"
    else:
        reference = spec.url
    fields: dict[str, Any] = {
        "source_kind": spec.source_kind,
        "reference": reference,
        "requested_revision": None,
        "resolved_revision": SHA,
        "access": "available",
        "access_guidance": None,
        "content": "images",
        "assets": (planned("DIV2K_valid_HR.zip"),),
        "supports_pause": True,
        "declared_splits": False,
        "split_mapping": {},
        "member_split_labels": {},
        "warnings": ("one warning",),
        "terms_reference": spec.terms_reference,
    }
    return ResolvedSource(**{**fields, **overrides})


def test_asset_identity_is_deterministic_and_sensitive_to_every_field() -> None:
    """Each pinned field changes the identity; equal inputs give equal names."""
    base = ("https_archive", "https://example.test/a.zip", '"e1"', "a.zip", None)
    identity = asset_identity(*base)
    assert identity == asset_identity(*base) and re.fullmatch(r"[a-f0-9]{64}", identity)
    variants = [
        ("hugging_face", *base[1:]),
        (base[0], "https://example.test/b.zip", *base[2:]),
        (*base[:2], '"e2"', *base[3:]),
        (*base[:3], "b.zip", None),
        (*base[:4], DIGEST),
    ]
    assert len({asset_identity(*variant) for variant in variants} | {identity}) == 6


def test_materialization_identity_tracks_selection_pin_and_image_cap() -> None:
    """Filters, pins, revisions and the image cap change the identity.

    Byte limits and the terms reference do not, because they never change
    which files land in the raw folder.
    """
    spec = hub_spec()
    source = resolved(spec)
    identity = materialization_identity(spec, source)
    assert identity == materialization_identity(spec, resolved(spec))
    assert identity == materialization_identity(spec, source, maximum_images=200_000)
    assert identity != materialization_identity(spec, source, maximum_images=3)
    same = [
        hub_spec(maximum_download_bytes=1024),
        hub_spec(terms_reference="https://elsewhere.test"),
    ]
    assert {materialization_identity(other, source) for other in same} == {identity}
    different = [
        hub_spec(path_prefix="data"),
        hub_spec(split="train"),
        hub_spec(file_names=["a.parquet"]),
        hub_spec(maximum_files=5),
        hub_spec(content="text"),
        hub_spec(archive_splits={"train": "train"}),
    ]
    others = {materialization_identity(other, source) for other in different}
    assert len(others) == len(different) and identity not in others
    assert (
        materialization_identity(spec, resolved(spec, resolved_revision="2" * 40))
        != identity
    )
    assert materialization_identity(spec, resolved(spec, reference="x")) != identity
    archive = archive_spec()
    plain = materialization_identity(archive, resolved(archive))
    assert (
        materialization_identity(
            archive_spec(expected_sha256=DIGEST), resolved(archive)
        )
        != plain
    )
    assert (
        materialization_identity(
            archive_spec(archive_splits={"a": "train"}), resolved(archive)
        )
        != plain
    )


def test_upstream_label_examples() -> None:
    """Folder names become safe lower-case counter keys."""
    assert upstream_label("DIV2K_valid_HR") == "div2k_valid_hr"
    assert upstream_label("val2017") == "val2017"
    assert upstream_label("1abc") == "label_1abc"
    assert upstream_label("Träin-Set") == "tr_in_set"
    assert upstream_label("") == "label_"
    assert len(upstream_label("x" * 100)) == 64
    for name in ("DIV2K_valid_HR", "1abc", "", "-", "x" * 100, "2017"):
        assert COUNTER_KEY.fullmatch(upstream_label(name)), name


def test_plan_for_maps_archive_splits_onto_labels() -> None:
    """Request folders become labels and assigned splits on the resolved source."""
    spec = archive_spec(
        archive_splits={"DIV2K_valid_HR": "held_out", "DIV2K_train_HR": "train"}
    )
    source = resolved(
        spec, split_mapping={"extra": "tuning"}, member_split_labels={"e": "extra"}
    )
    plan = plan_for(
        spec, source, source_name="div2k", maximum_images=5, training_intended=False
    )
    assert plan.resolved.split_mapping == {
        "extra": "tuning",
        "div2k_valid_hr": "held_out",
        "div2k_train_hr": "train",
    }
    assert plan.resolved.member_split_labels == {
        "e": "extra",
        "DIV2K_valid_HR": "div2k_valid_hr",
        "DIV2K_train_HR": "div2k_train_hr",
    }
    assert plan.resolved.declared_splits is True
    assert plan.identity == materialization_identity(spec, source, maximum_images=5)
    assert (plan.source_name, plan.maximum_images, plan.training_intended) == (
        "div2k",
        5,
        False,
    )
    bare = plan_for(
        archive_spec(),
        resolved(archive_spec()),
        source_name="d",
        maximum_images=1,
        training_intended=True,
    )
    assert bare.resolved.split_mapping == {} and bare.resolved.declared_splits is False
    conflicting = archive_spec(archive_splits={"Val": "tuning", "val": "held_out"})
    with pytest.raises(ApplicationFailure) as failure:
        plan_for(
            conflicting,
            resolved(conflicting),
            source_name="d",
            maximum_images=1,
            training_intended=True,
        )
    assert failure.value.code == "source_split_labels"


def test_suggested_source_names() -> None:
    """Names come from the repository, the archive file or the upload name."""
    assert suggested_source_name(resolved(hub_spec())) == "my-data"
    assert suggested_source_name(resolved(archive_spec())) == "div2k_valid_hr"
    upload = UploadSourceSpec(upload_identifier="upload_" + "0" * 32)
    assert suggested_source_name(resolved(upload)) == "photos_set"
    assert (
        suggested_source_name(
            resolved(archive_spec(), reference="https://x.test/1.zip")
        )
        == "source"
    )


def test_inspection_reports_cache_and_disk_honestly(tmp_path: Path) -> None:
    """Cached bytes, totals and disk sufficiency follow the resolved assets."""
    cache = DatasetCache(tmp_path / "cache")
    data_root = tmp_path / "data"
    data_root.mkdir()
    spec = archive_spec()
    assets = (planned("DIV2K_valid_HR.zip", 100), planned("extra.zip", 50))
    source = resolved(spec, assets=assets)
    local = tmp_path / "local.zip"
    local.write_bytes(bytes(100))
    cache.import_file(
        "https_archive",
        asset_identity("https_archive", source.reference, SHA, assets[0].path, None),
        assets[0],
        local,
        owner="job_a",
        reference=source.reference,
        revision=SHA,
        terms_reference=spec.terms_reference,
        created_at="2026-09-27T00:00:00+00:00",
    )
    inspection = inspection_for(
        spec,
        source,
        source_name=None,
        cache=cache,
        data_root=data_root,
        token_configured=True,
    )
    assert inspection.suggested_source_name == "div2k_valid_hr"
    assert (inspection.download_bytes, inspection.materialized_bytes) == (150, 165)
    assert inspection.cached_bytes == 100
    assert inspection.required_free_bytes == 50 + 165 + minimum_free_bytes(data_root)
    assert inspection.disk_sufficient == (
        free_disk_bytes(data_root) >= inspection.required_free_bytes
    )
    assert [asset.path for asset in inspection.assets] == [
        "DIV2K_valid_HR.zip",
        "extra.zip",
    ]
    assert inspection.materialization_identity == materialization_identity(spec, source)
    assert inspection.raw_folder in (None, "available")
    assert inspection.server_token_configured is True
    assert inspection.warnings == ["one warning"]
    unknown = replace(source, assets=(planned("DIV2K_valid_HR.zip", None),))
    partial = inspection_for(
        spec,
        unknown,
        source_name="named",
        cache=cache,
        data_root=data_root,
        token_configured=False,
    )
    assert partial.suggested_source_name == "named"
    assert (
        partial.download_bytes,
        partial.required_free_bytes,
        partial.disk_sufficient,
    ) == (None, None, None)
    # The asset identity ignores size, so the cache still knows this file.
    assert partial.cached_bytes == 100
