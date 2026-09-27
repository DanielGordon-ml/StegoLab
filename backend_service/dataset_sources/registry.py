"""Find the adapter that describes each kind of dataset source."""

import importlib
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, cast

from backend_service.dataset_sources.plans import ResolvedSource
from backend_service.dataset_sources.transport import SecureTransport
from backend_service.failures import ApplicationFailure
from schemas.dataset_sources import DatasetSourceSpec

if TYPE_CHECKING:
    from backend_service.dataset_sources.upload_sessions import DatasetUploadStore

logger = logging.getLogger(__name__)

TransportFactory = Callable[[dict[str, str] | None], SecureTransport]
ADAPTER_MODULES: dict[str, str] = {
    "hugging_face": "backend_service.dataset_sources.hugging_face",
    "https_archive": "backend_service.dataset_sources.https_archive",
    "upload": "backend_service.dataset_sources.upload_source",
}
_MESSAGES = {
    "source_kind_unsupported": (
        "This kind of dataset source is not supported. Use a Hugging Face "
        "dataset, an https archive address or an uploaded archive."
    ),
    "source_adapter_unavailable": (
        "The support module for this kind of dataset source is missing or "
        "incomplete in this installation. Reinstall the backend service."
    ),
}


def registry_failure(code: str) -> ApplicationFailure:
    """Return the fixed message for a source registry failure code."""
    status_code = 422 if code == "source_kind_unsupported" else 501
    return ApplicationFailure(code, _MESSAGES[code], status_code)


class SourceAdapter(Protocol):
    """Describe a dataset source before any bytes are downloaded."""

    def resolve(
        self,
        spec: DatasetSourceSpec,
        *,
        transport_factory: TransportFactory,
        uploads: "DatasetUploadStore | None",
    ) -> ResolvedSource:
        """Return what the source holds, how it is versioned and if it can pause."""
        ...


def adapter_for(spec: DatasetSourceSpec) -> SourceAdapter:
    """Load the adapter module for the source kind the spec declares.

    Modules load on first use so one missing adapter never blocks the others.
    """
    module_name = ADAPTER_MODULES.get(spec.source_kind)
    if module_name is None:
        raise registry_failure("source_kind_unsupported")
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        logger.error("source_adapter_import_failed")
        raise registry_failure("source_adapter_unavailable") from None
    if not callable(getattr(module, "resolve", None)):
        logger.error("source_adapter_resolve_missing")
        raise registry_failure("source_adapter_unavailable")
    return cast(SourceAdapter, module)
