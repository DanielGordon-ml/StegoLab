"""Cached downloads become published raw folders with metadata and markers."""

import hashlib
import io
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pyarrow
import pyarrow.parquet
import pytest
from archive_fixtures import ZipMember, build_zip
from PIL import Image

from backend_service.dataset_preparation import prepare_dataset
from backend_service.dataset_sources import materialization, plans, source_marker
from backend_service.dataset_sources.cache import DatasetCache
from backend_service.dataset_sources.transfer import DownloadStopped
from backend_service.dataset_validation import load_manifest, validate_dataset
from backend_service.failures import ApplicationFailure
from schemas.dataset_sources import HttpsArchiveSourceSpec, HuggingFaceSourceSpec
from schemas.datasets import DatasetPreparationRequest

Plan = plans.MaterializationPlan
Source = materialization.MaterializedSource
SHA = "2" * 40
CREATED_AT = "2026-09-27T00:00:00+00:00"
TERMS = "https://example.test/terms"
URL = "https://example.test/div2k.zip"
HUB = f"https://huggingface.co/datasets/fake/pictures/tree/{SHA}"
SHARD = "data/train-00000-of-00001.parquet"
METADATA = source_marker.METADATA_NAME
IMAGE_RECORD = pyarrow.struct([("bytes", pyarrow.binary()), ("path", pyarrow.string())])
IMPORT_OPTIONS = {"owner": "job_a", "terms_reference": TERMS, "created_at": CREATED_AT}
ARCHIVE_SPLITS = {"DIV2K_train_HR": "train", "DIV2K_valid_HR": "held_out"}
HUB_SPLITS = {"train": "train", "validation": "tuning"}
DIV2K_MEMBERS = (
    "DIV2K_train_HR/0001.png",
    "DIV2K_train_HR/0002.png",
    "DIV2K_valid_HR/0801.png",
    "DIV2K_valid_HR/0802.png",
)
RESOLVED_DEFAULTS: dict[str, Any] = dict(
    access="available", content="images", supports_pause=True, declared_splits=False
)
RESOLVED_DEFAULTS.update(
    requested_revision=None, access_guidance=None, split_mapping={}, warnings=()
)
RESOLVED_DEFAULTS.update(member_split_labels={}, terms_reference=TERMS)


def smooth_png(seed: int) -> bytes:
    """Return a real 256x256 PNG whose gradient differs per seed."""
    gradient = Image.linear_gradient("L")
    shifted = gradient.point(lambda value: (value + seed * 40) % 256)
    buffer = io.BytesIO()
    Image.merge("RGB", (gradient, shifted, gradient.rotate(90))).save(buffer, "PNG")
    return buffer.getvalue()


def div2k_zip(members: tuple[str, ...] = DIV2K_MEMBERS) -> dict[str, bytes]:
    """Build a DIV2K-shaped zip plus one unsupported file, keyed by asset path."""
    files = {name: smooth_png(index) for index, name in enumerate(members)}
    entries = [ZipMember(name, data) for name, data in files.items()]
    return {"div2k.zip": build_zip([*entries, ZipMember("README.md", b"junk")])}


def shard(column: str, values: Any) -> bytes:
    """Write one parquet shard with a single column and return its bytes."""
    buffer = pyarrow.BufferOutputStream()
    table = pyarrow.table({column: values})
    pyarrow.parquet.write_table(table, buffer, row_group_size=2)
    return bytes(buffer.getvalue().to_pybytes())


def image_shard(rows: list[bytes]) -> bytes:
    """Encode image rows the way Hugging Face image datasets do."""
    records = [{"bytes": data, "path": ""} for data in rows]
    return shard("image", pyarrow.array(records, type=IMAGE_RECORD))


def planned(path: str, data: bytes) -> plans.PlannedAsset:
    """Describe one asset whose bytes the tests place in the cache."""
    digest = hashlib.sha256(data).hexdigest()
    return plans.PlannedAsset(URL, path, len(data), digest, None, True, None)


def planned_source(spec: Any, name: str, data: dict[str, bytes], **fields: Any) -> Plan:
    """Resolve the spec the way an adapter would and bind it to a folder name."""
    options: dict[str, Any] = {"training_intended": True}
    options["maximum_images"] = fields.pop("maximum_images", 200_000)
    resolved = plans.ResolvedSource(
        assets=tuple(planned(path, content) for path, content in data.items()),
        **{**RESOLVED_DEFAULTS, **fields},
    )
    return plans.plan_for(spec, resolved, source_name=name, **options)


def archive_plan(assets: dict[str, bytes], **fields: Any) -> Plan:
    """Plan an https archive fetch; ``splits`` overrides the folder mapping."""
    splits = fields.pop("splits", ARCHIVE_SPLITS)
    spec = HttpsArchiveSourceSpec.model_validate(
        {"url": URL, "terms_reference": TERMS, "archive_splits": splits}
    )
    fields.update(source_kind="https_archive", reference=URL, resolved_revision="etag")
    return planned_source(spec, "div2k_sample", assets, **fields)


def hub_plan(shards: dict[str, bytes], **fields: Any) -> Plan:
    """Plan a Hugging Face fetch of parquet shards labelled by their split name."""
    content = fields.get("content", "images")
    spec = HuggingFaceSourceSpec.model_validate(
        {"repository": "fake/pictures", "content": content, "terms_reference": TERMS}
    )
    labels = {path.rsplit("/", 1)[-1]: "train" for path in shards}
    hub: dict[str, Any] = {"source_kind": "hugging_face", "reference": HUB}
    hub.update(resolved_revision=SHA, declared_splits=True, member_split_labels=labels)
    hub["split_mapping"] = HUB_SPLITS
    return planned_source(spec, "pictures", shards, **hub, **fields)


def materialized(
    root: Path, plan: Plan, data: dict[str, bytes], **options: Any
) -> Source:
    """Cache the given asset bytes as complete entries, then materialize the plan."""
    data_root, cache_root = root / "workspace" / "data", root / "cache"
    data_root.mkdir(parents=True, exist_ok=True)
    source, cache, entries = plan.resolved, DatasetCache(cache_root), {}
    for asset in source.assets:
        if asset.path not in data:
            continue
        saved = root / "downloads" / asset.basename
        saved.parent.mkdir(exist_ok=True)
        saved.write_bytes(data[asset.path])
        pinned = (source.source_kind, source.reference, source.resolved_revision)
        identity = plans.asset_identity(*pinned, asset.path, asset.expected_sha256)
        publish = {"reference": source.reference, "revision": source.resolved_revision}
        entries[identity] = cache.import_file(
            source.source_kind, identity, asset, saved, **publish, **IMPORT_OPTIONS
        )
    where: dict[str, Any] = {"created_at": CREATED_AT, "data_root": data_root}
    where["cache_root"] = cache_root
    with pytest.MonkeyPatch.context() as patch:
        for name in ("mkdtemp", "mkstemp", "TemporaryDirectory", "NamedTemporaryFile"):
            patch.setattr(tempfile, name, refuse_temporary_files)
        return materialization.materialize(plan, entries, **where, **options)


def refuse_temporary_files(*arguments: Any, **options: Any) -> Any:
    """Fail any attempt to create a temporary file or folder."""
    raise AssertionError("Materialization must not use temporary files.")


@pytest.fixture(autouse=True)
def run_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep run reports inside the test folder."""
    monkeypatch.setenv("STEGOLAB_LOG_DIRECTORY", str(tmp_path / "logs"))


def test_archive_members_land_with_metadata_marker_and_reuse(tmp_path: Path) -> None:
    """Members keep their paths, the CSV and marker are written, then reused."""
    assets = div2k_zip()
    plan, data_root = archive_plan(assets), tmp_path / "workspace" / "data"
    reports: list[int] = []
    before = set(os.listdir("/tmp"))
    result = materialized(
        tmp_path, plan, assets, progress=lambda done, total: reports.append(done)
    )
    folder = result.directory
    assert folder == data_root / "div2k_sample" and not result.reused
    for index, name in enumerate(DIV2K_MEMBERS):
        assert (folder / name).read_bytes() == smooth_png(index)
    rows = (folder / METADATA).read_text().splitlines()
    assert rows[0] == "image_name,set,subset,identity" and len(rows) == 5
    assert rows[3] == (
        "0801.png,div2k_valid_hr,DIV2K_valid_HR,div2k_sample:DIV2K_valid_HR/0801.png"
    )
    marker = source_marker.read_source_marker(folder)
    assert marker == result.marker and marker.completed and marker.declared_splits
    assert (marker.member_count, marker.rejected_member_count) == (4, 1)
    assert marker.materialization_identity == plan.identity
    expected = {"div2k_train_hr": "train", "div2k_valid_hr": "held_out"}
    assert marker.split_mapping == expected == plan.resolved.split_mapping
    assert marker.assets[0].sha256 == hashlib.sha256(assets["div2k.zip"]).hexdigest()
    assert result.rejection_reasons == {"unsupported_file_type": 1}
    assert not result.warnings
    assert reports[-1] == sum((folder / name).stat().st_size for name in DIV2K_MEMBERS)
    assert not list((data_root / ".staging").iterdir())
    assert not [name for name in set(os.listdir("/tmp")) - before if "div2k" in name]
    reused = materialization.find_materialized(data_root, plan)
    assert reused is not None and reused.reused and reused.marker == marker
    again = materialized(tmp_path, plan, {})
    assert again.reused and again.marker == marker
    with pytest.raises(ApplicationFailure) as failure:
        materialization.find_materialized(data_root, archive_plan(assets, splits=None))
    assert failure.value.code == "source_folder_occupied"
    assert failure.value.status_code == 409
    (data_root / "foreign").mkdir()
    (data_root / "foreign" / "one.png").write_bytes(b"x")
    with pytest.raises(ApplicationFailure) as failure:
        materialized(tmp_path, replace(plan, source_name="foreign"), assets)
    assert failure.value.code == "source_folder_occupied"


def test_published_folder_prepares_with_provenance(tmp_path: Path) -> None:
    """The preparation document from the marker yields a revision with provenance."""
    assets = div2k_zip()
    plan = archive_plan(assets)
    result = materialized(tmp_path, plan, assets)
    output = tmp_path / "workspace" / "datasets"
    document = source_marker.preparation_document(result.directory, "div2k", output)
    summary = prepare_dataset(DatasetPreparationRequest.model_validate(document))
    directory = output / "div2k" / summary.revision
    manifest = load_manifest(directory)
    assert (summary.accepted_count, summary.rejection_count) == (4, 0)
    assert summary.unique_eligible_by_split == {"train": 2, "tuning": 0, "held_out": 2}
    assert manifest.split_mapping == plan.resolved.split_mapping
    assert manifest.source_provenance["source_name"] == "div2k_sample"
    assert manifest.source_provenance["source_materialization"] == plan.identity
    assert manifest.source_provenance["source_revision"] == "etag"
    assert manifest.metadata_checksum is not None
    assert validate_dataset(directory).integrity == "verified"


def test_parquet_rows_become_hashed_images_with_row_identities(tmp_path: Path) -> None:
    """Image rows are saved by checksum under their label; junk and repeats skip."""
    rows = [smooth_png(1), smooth_png(2), b"not an image", smooth_png(1)]
    shards = {SHARD: image_shard(rows)}
    result = materialized(tmp_path, hub_plan(shards), shards)
    digests = [hashlib.sha256(data).hexdigest() for data in rows[:2]]
    saved = sorted(path.name for path in (result.directory / "train").iterdir())
    assert saved == sorted(f"{digest}.png" for digest in digests)
    assert (result.directory / "train" / f"{digests[1]}.png").read_bytes() == rows[1]
    lines = (result.directory / METADATA).read_text().splitlines()
    identity = f"fake/pictures@{SHA}:{SHARD}#0"
    assert f"{digests[0]}.png,train,train-00000-of-00001.parquet,{identity}" in lines
    skipped = {"duplicate_member": 1, "unsupported_file_type": 1}
    assert result.rejection_reasons == skipped
    assert result.marker.split_mapping == HUB_SPLITS and result.marker.declared_splits
    assert (result.marker.member_count, result.marker.rejected_member_count) == (2, 2)


def test_text_rows_are_saved_per_label_with_a_corpus_summary(tmp_path: Path) -> None:
    """Text shards of one label join into one file summarized without content."""
    shards = {
        "data/train-00000-of-00002.parquet": shard("text", ["alpha", "bêta"]),
        "data/train-00001-of-00002.parquet": shard("text", ["gamma"]),
    }
    result = materialized(tmp_path, hub_plan(shards, content="text"), shards)
    saved = result.directory / "data" / "train.txt"
    assert saved.read_text(encoding="utf-8") == "alpha\nbêta\ngamma"
    assert result.text_corpus is not None and result.text_corpus.files == 1
    assert (result.text_corpus.characters, result.text_corpus.bytes) == (16, 17)
    assert not (result.directory / METADATA).exists()
    assert result.marker.content == "text" and not result.marker.declared_splits
    assert (result.marker.member_count, result.marker.split_mapping) == (1, {})


def test_image_cap_and_unlabelled_images_shape_the_metadata(tmp_path: Path) -> None:
    """Images past the cap are rejected; an unlabelled image drops the CSV."""
    assets = div2k_zip()
    result = materialized(tmp_path, archive_plan(assets, maximum_images=3), assets)
    assert not (result.directory / "DIV2K_valid_HR" / "0802.png").exists()
    assert all((result.directory / name).exists() for name in DIV2K_MEMBERS[:3])
    assert len((result.directory / METADATA).read_text().splitlines()) == 4
    assert result.rejection_reasons == {"image_limit": 1, "unsupported_file_type": 1}
    shards = {SHARD: image_shard([smooth_png(3), smooth_png(4)])}
    capped = materialized(tmp_path, hub_plan(shards, maximum_images=1), shards)
    assert capped.rejection_reasons == {"image_limit": 1}
    assert (result.marker.member_count, capped.marker.member_count) == (3, 1)
    assets = div2k_zip((*DIV2K_MEMBERS, "0900.png"))
    mixed = replace(archive_plan(assets), source_name="mixed")
    loose = materialized(tmp_path, mixed, assets)
    assert (loose.directory / "0900.png").exists()
    assert not (loose.directory / METADATA).exists()
    assert not loose.marker.declared_splits and loose.marker.split_mapping == {}
    assert loose.warnings and "not saved" in loose.warnings[0]
    document = source_marker.preparation_document(loose.directory, "plain", tmp_path)
    assert document["metadata_file"] is None and document["split_mapping"] == {}


def test_stop_between_members_leaves_no_folder(tmp_path: Path) -> None:
    """A stop request between members raises and removes the stage."""
    assets = div2k_zip()
    plan, ticks = archive_plan(assets), iter(range(10))
    with pytest.raises(DownloadStopped):
        materialized(tmp_path, plan, assets, stop=lambda: next(ticks) > 1)
    data_root = tmp_path / "workspace" / "data"
    assert not (data_root / "div2k_sample").exists()
    assert not list((data_root / ".staging").iterdir())
    assert materialization.find_materialized(data_root, plan) is None
