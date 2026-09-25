"""Bounded, expiring storage of uploads and results for experimental inference."""

import hashlib
import os
import re
import shutil
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from backend_service.failures import ApplicationFailure, StorageFailure
from backend_service.image_preparation import prepare_image, read_prepared_png
from backend_service.image_validation import PNG_SIGNATURE
from backend_service.inference_storage import (
    directory_bytes,
    read_regular_file,
    summary_warnings,
    upload_failure,
    write_new_file,
)
from backend_service.workspace_registry import safe_path
from schemas.capabilities import MAXIMUM_UPLOAD_BYTES
from schemas.images import ImageSummary
from schemas.inference import (
    MAXIMUM_STORED_IMAGE_BYTES,
    RETENTION_SECONDS,
    ImagePurpose,
    StoredImageRecord,
    UploadedImage,
)

STORAGE_CAP_BYTES = 512 * 1024**2
SCRATCH_RETENTION_SECONDS = 60 * 60
IMAGE_FILE = "image.png"
RECORD_FILE = "record.json"
STORED_REFERENCE = re.compile(r"^(image|encoded)_[0-9a-f]{32}$")


class InferenceFileStore:
    """Keep uploads and published results under the state directory for one day."""

    def __init__(
        self, directory: Path, clock: Callable[[], datetime] | None = None
    ) -> None:
        """Create private upload, result, and scratch directories."""
        root = directory / "inference"
        self.images = root / "images"
        self.results = root / "results"
        self.scratch = root / "scratch"
        self.clock = clock or (lambda: datetime.now(UTC))
        self.lock = threading.Lock()
        try:
            for path in (self.images, self.results, self.scratch):
                path.mkdir(parents=True, exist_ok=True, mode=0o700)
        except OSError:
            raise StorageFailure() from None

    def _directory(self, reference: str) -> Path:
        """Return the contained directory for a well-formed stored reference."""
        try:
            if not STORED_REFERENCE.fullmatch(reference):
                raise ValueError
            parent = self.images if reference.startswith("image_") else self.results
            return safe_path(parent / reference, parent)
        except (OSError, ValueError):
            raise upload_failure("image_expired") from None

    def used_bytes(self) -> int:
        """Sum the regular files kept under the upload and result directories."""
        return directory_bytes(self.images) + directory_bytes(self.results)

    def store(self, purpose: ImagePurpose, data: bytes) -> UploadedImage:
        """Validate, save, and describe one upload inside the retention window."""
        if len(data) > MAXIMUM_UPLOAD_BYTES:
            raise upload_failure("upload_too_large")
        with self.lock:
            self.sweep()
            try:
                if self.used_bytes() + len(data) > STORAGE_CAP_BYTES:
                    raise upload_failure("inference_storage_full")
                identifier = "image_" + uuid4().hex
                directory = self.images / identifier
                directory.mkdir(mode=0o700)
            except OSError:
                raise StorageFailure() from None
            try:
                return self._write(identifier, directory, purpose, data).public()
            except BaseException:
                shutil.rmtree(directory, ignore_errors=True)
                raise

    def _write(
        self, identifier: str, directory: Path, purpose: ImagePurpose, data: bytes
    ) -> StoredImageRecord:
        """Save the prepared cover or the exact encoded PNG with its record."""
        image_path = directory / IMAGE_FILE
        try:
            if purpose == "cover":
                summary = prepare_image(BytesIO(data), image_path)
                stored = read_regular_file(image_path, MAXIMUM_STORED_IMAGE_BYTES)
                warnings = summary_warnings(summary)
            else:
                if not data.startswith(PNG_SIGNATURE):
                    raise upload_failure("image_purpose_format")
                summary = read_prepared_png(BytesIO(data)).summary
                write_new_file(image_path, data)
                stored, warnings = data, []
            if self.used_bytes() > STORAGE_CAP_BYTES:
                raise upload_failure("inference_storage_full")
            return self._record(
                identifier, directory, purpose, summary, stored, warnings
            )
        except (OSError, ValueError):
            raise StorageFailure() from None

    def _record(
        self,
        identifier: str,
        directory: Path,
        purpose: ImagePurpose,
        summary: ImageSummary,
        stored: bytes,
        warnings: list[str],
    ) -> StoredImageRecord:
        """Write the JSON record next to a stored PNG that was already checked."""
        now = self.clock()
        record = StoredImageRecord(
            image_reference=identifier,
            purpose=purpose,
            summary=summary,
            warnings=warnings,
            created_at=now,
            expires_at=now + timedelta(seconds=RETENTION_SECONDS),
            stored_bytes=len(stored),
            checksum=hashlib.sha256(stored).hexdigest(),
        )
        write_new_file(
            directory / RECORD_FILE, record.model_dump_json().encode("utf-8")
        )
        return record

    def publish_result(self, source: Path) -> StoredImageRecord:
        """Publish a verified PNG produced in scratch as an expiring result."""
        with self.lock:
            self.sweep()
            try:
                data = read_regular_file(source, MAXIMUM_STORED_IMAGE_BYTES)
                if self.used_bytes() + len(data) > STORAGE_CAP_BYTES:
                    raise upload_failure("inference_storage_full")
                identifier = "encoded_" + uuid4().hex
                directory = self.results / identifier
                directory.mkdir(mode=0o700)
            except (OSError, ValueError):
                raise StorageFailure() from None
            try:
                summary = read_prepared_png(BytesIO(data)).summary
                try:
                    os.link(source, directory / IMAGE_FILE)
                except OSError:
                    write_new_file(directory / IMAGE_FILE, data)
                return self._record(identifier, directory, "encoded", summary, data, [])
            except BaseException:
                shutil.rmtree(directory, ignore_errors=True)
                raise

    def read_record(self, reference: str) -> StoredImageRecord:
        """Return one unexpired stored record or a safe not-found failure."""
        directory = self._directory(reference)
        try:
            record = StoredImageRecord.model_validate_json(
                read_regular_file(directory / RECORD_FILE, 64 * 1024)
            )
        except (OSError, ValueError, ValidationError):
            raise upload_failure("image_expired") from None
        if record.image_reference != reference or record.expires_at <= self.clock():
            raise upload_failure("image_expired")
        return record

    def image_path(self, reference: str) -> Path:
        """Return the stored PNG path of an unexpired record for server-side use."""
        self.read_record(reference)
        return self._directory(reference) / IMAGE_FILE

    def read_image(self, reference: str) -> tuple[bytes, str]:
        """Return exact stored PNG bytes after confirming the recorded checksum."""
        try:
            record = self.read_record(reference)
            content = read_regular_file(
                self._directory(reference) / IMAGE_FILE, MAXIMUM_STORED_IMAGE_BYTES
            )
            if (
                len(content) != record.stored_bytes
                or hashlib.sha256(content).hexdigest() != record.checksum
            ):
                raise ValueError
        except (OSError, ValueError, ApplicationFailure):
            raise upload_failure("artifact_expired") from None
        short = reference.split("_", 1)[1][:8]
        return content, f"stegolab-{record.purpose}-{short}.png"

    def sweep(self) -> int:
        """Delete expired records, stale scratch folders, and count removals."""
        removed = 0
        now = self.clock()
        try:
            for parent in (self.images, self.results):
                for directory in list(parent.iterdir()):
                    try:
                        if directory.is_symlink() or not directory.is_dir():
                            continue
                        expired = self._expired(directory, now)
                    except FileNotFoundError:
                        continue
                    if expired:
                        shutil.rmtree(directory, ignore_errors=True)
                        removed += 1
            for directory in list(self.scratch.iterdir()):
                try:
                    age = now.timestamp() - directory.lstat().st_mtime
                except FileNotFoundError:
                    continue
                if not directory.is_symlink() and age >= SCRATCH_RETENTION_SECONDS:
                    shutil.rmtree(directory, ignore_errors=True)
                    removed += 1
        except OSError:
            raise StorageFailure() from None
        return removed

    def _expired(self, directory: Path, now: datetime) -> bool:
        """Expire by the record, or by age when the record cannot be read."""
        try:
            record = StoredImageRecord.model_validate_json(
                read_regular_file(directory / RECORD_FILE, 64 * 1024)
            )
            return record.expires_at <= now
        except (OSError, ValueError, ValidationError):
            age = now.timestamp() - directory.lstat().st_mtime
            return age >= RETENTION_SECONDS
