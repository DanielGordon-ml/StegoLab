"""Fetch remote dataset files over checked https connections."""

from backend_service.dataset_sources.address_policy import AddressPolicy
from backend_service.dataset_sources.fetch import fetch_dataset, inspect_source
from backend_service.dataset_sources.progress import FetchProgressSink
from backend_service.dataset_sources.transfer import (
    DownloadStopped,
    RemoteAsset,
    download_asset,
)
from backend_service.dataset_sources.transport import SecureTransport

__all__ = [
    "AddressPolicy",
    "DownloadStopped",
    "FetchProgressSink",
    "RemoteAsset",
    "SecureTransport",
    "download_asset",
    "fetch_dataset",
    "inspect_source",
]
