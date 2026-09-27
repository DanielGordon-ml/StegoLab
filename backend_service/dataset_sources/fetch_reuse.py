"""Reuse the raw folder of an earlier fetch of the same source without network."""

import logging
import re
from pathlib import Path

from backend_service.dataset_sources.https_archive import archive_reference
from backend_service.dataset_sources.hugging_face_files import HUB_HOST
from backend_service.dataset_sources.plans import (
    ResolvedSource,
    materialization_identity,
)
from backend_service.dataset_sources.source_marker import read_source_marker
from backend_service.failures import ApplicationFailure
from schemas.dataset_sources import (
    MAXIMUM_IMAGES,
    DatasetSourceSpec,
    HttpsArchiveSourceSpec,
    HuggingFaceSourceSpec,
    SourceMarker,
)

logger = logging.getLogger(__name__)

COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
REUSE_WARNING = (
    "The raw folder from an earlier fetch of this source was reused without "
    "asking the server whether the source changed. Inspect the source to check "
    "for a newer version."
)


def reference_matches(spec: DatasetSourceSpec, marker: SourceMarker) -> bool:
    """Tell whether a marker describes the source this spec points at.

    Archives match by their address without its query. Hub repositories match
    by repository; a request that pins a commit must pin the marker's commit.
    Uploads never match, because they are resolved locally anyway.
    """
    if isinstance(spec, HttpsArchiveSourceSpec):
        return marker.reference == archive_reference(spec.url)
    if isinstance(spec, HuggingFaceSourceSpec):
        prefix = f"https://{HUB_HOST}/datasets/{spec.repository}/tree/"
        pinned = COMMIT_SHA.fullmatch(spec.revision) is not None
        return marker.reference == prefix + marker.resolved_revision and (
            not pinned or spec.revision == marker.resolved_revision
        )
    return False


def offline_source(marker: SourceMarker) -> ResolvedSource:
    """Describe a source from its marker alone, without any assets to fetch."""
    return ResolvedSource(
        source_kind=marker.source_kind,
        reference=marker.reference,
        requested_revision=None,
        resolved_revision=marker.resolved_revision,
        access="available",
        access_guidance=None,
        content=marker.content,
        assets=(),
        supports_pause=False,
        declared_splits=marker.declared_splits,
        split_mapping=dict(marker.split_mapping),
        member_split_labels={},
        warnings=(REUSE_WARNING,),
        terms_reference=marker.terms_reference,
    )


def reusable_source(
    spec: DatasetSourceSpec,
    data_root: Path,
    source_name: str,
    *,
    maximum_images: int = MAXIMUM_IMAGES,
) -> ResolvedSource | None:
    """Describe the source from a completed marker that this exact request made.

    The marker must name the same source, and the request's selection and
    image cap at the marker's revision must give the marker's materialization
    identity. The fetch then needs no network at all; its summary carries a
    warning saying that the server was not asked for a newer version.
    """
    try:
        marker = read_source_marker(data_root / source_name)
    except ApplicationFailure:
        return None
    if (
        marker is None
        or not marker.completed
        or marker.source_kind != spec.source_kind
        or not reference_matches(spec, marker)
    ):
        return None
    offline = offline_source(marker)
    identity = materialization_identity(spec, offline, maximum_images=maximum_images)
    if identity != marker.materialization_identity:
        return None
    logger.info("fetch_source_reused_offline")
    return offline
