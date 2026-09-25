"""Bounded, expiring storage of uploaded images for experimental encode and decode."""

import hashlib
import os
import re
import shutil
import stat
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
IMAGE_FILE = "image.png"
RECORD_FILE = "record.json"
IMAGE_REFERENCE = re.compile(r"^image_[0-9a-f]{32}$")


def upload_failure(code: str) -> ApplicationFailure:
    """Return a fixed upload failure without file names, sizes, or bytes."""
    messages = {
        "upload_too_large": (
            413,
            "The image is larger than the 16 MiB upload limit. "
            "Choose a smaller file, then retry.",
        ),
        "upload_content_type": (
            415,
            "Send the image file as the request body with the content type "
            "image/png or image/jpeg.",
        ),
        "image_purpose_format": (
            422,
            "Decode needs the PNG file saved by Encode, unchanged. Converted or "
            "re-saved files, including JPEG, cannot be decoded.",
        ),
        "image_expired": (
            404,
            "This uploaded image is no longer available. Upload it again.",
        ),
        "artifact_expired": (
            404,
            "This image is no longer available. Upload or encode it again.",
        ),
        "inference_storage_full": (
            503,
            "Local image storage is full. Wait for older uploads to expire, "
            "then retry.",
        ),
    }
    status_code, message = messages[code]
    return ApplicationFailure(code, message, status_code)


def summary_warnings(summary: ImageSummary) -> list[str]:
    """Describe preparation steps the user should know about, in fixed words."""
    warnings: list[str] = []
    if summary.source_format == "JPEG":
        warnings.append(
            "The JPEG file was converted to PNG. Encode and Decode always use PNG."
        )
    if summary.orientation != 1:
        warnings.append("The image was rotated to its displayed orientation.")
    if summary.color_policy == "icc_to_srgb":
        warnings.append("The image colors were converted to standard sRGB.")
    return warnings


def read_regular_file(path: Path, maximum: int) -> bytes:
    """Read a bounded regular file without following a final symlink."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        information = os.fstat(stream.fileno())
        if not stat.S_ISREG(information.st_mode) or information.st_size > maximum:
            raise ValueError("The stored file exceeds its size or type limit.")
        data = stream.read(maximum + 1)
        if len(data) != information.st_size:
            raise ValueError("The stored file changed while it was read.")
        return data


def write_new_file(path: Path, data: bytes) -> None:
    """Create a private file that must not exist yet and flush it to disk."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


class InferenceFileStore:
    """Keep uploads under the application state directory for one day at most."""

    def __init__(
        self, directory: Path, clock: Callable[[], datetime] | None = None
    ) -> None:
        """Create the private image directory inside the application state."""
        self.images = directory / "inference" / "images"
        self.clock = clock or (lambda: datetime.now(UTC))
        self.lock = threading.Lock()
        try:
            self.images.mkdir(parents=True, exist_ok=True, mode=0o700)
        except OSError:
            raise StorageFailure() from None

    def _directory(self, image_reference: str) -> Path:
        """Return the contained directory for a well-formed reference."""
        try:
            if not IMAGE_REFERENCE.fullmatch(image_reference):
                raise ValueError
            return safe_path(self.images / image_reference, self.images)
        except (OSError, ValueError):
            raise upload_failure("image_expired") from None

    def used_bytes(self) -> int:
        """Sum the regular files kept under the image directory."""
        total = 0
        for directory in self.images.iterdir():
            if directory.is_symlink() or not directory.is_dir():
                continue
            for child in directory.iterdir():
                information = child.lstat()
                if stat.S_ISREG(information.st_mode):
                    total += information.st_size
        return total

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
        except (OSError, ValueError):
            raise StorageFailure() from None

    def read_record(self, image_reference: str) -> StoredImageRecord:
        """Return one unexpired upload record or a safe not-found failure."""
        directory = self._directory(image_reference)
        try:
            record = StoredImageRecord.model_validate_json(
                read_regular_file(directory / RECORD_FILE, 64 * 1024)
            )
        except (OSError, ValueError, ValidationError):
            raise upload_failure("image_expired") from None
        if record.image_reference != image_reference or record.expires_at <= (
            self.clock()
        ):
            raise upload_failure("image_expired")
        return record

    def image_path(self, image_reference: str) -> Path:
        """Return the stored PNG path of an unexpired upload for server-side use."""
        self.read_record(image_reference)
        return self._directory(image_reference) / IMAGE_FILE

    def read_image(self, image_reference: str) -> tuple[bytes, str]:
        """Return exact stored PNG bytes after confirming the recorded checksum."""
        try:
            record = self.read_record(image_reference)
            content = read_regular_file(
                self._directory(image_reference) / IMAGE_FILE,
                MAXIMUM_STORED_IMAGE_BYTES,
            )
            if (
                len(content) != record.stored_bytes
                or hashlib.sha256(content).hexdigest() != record.checksum
            ):
                raise ValueError
        except (OSError, ValueError, ApplicationFailure):
            raise upload_failure("artifact_expired") from None
        return content, f"stegolab-{record.purpose}-{image_reference[6:14]}.png"

    def sweep(self) -> int:
        """Delete expired or unreadable upload directories and count them."""
        removed = 0
        now = self.clock()
        try:
            for directory in list(self.images.iterdir()):
                if directory.is_symlink() or not directory.is_dir():
                    continue
                try:
                    record = StoredImageRecord.model_validate_json(
                        read_regular_file(directory / RECORD_FILE, 64 * 1024)
                    )
                    expired = record.expires_at <= now
                except (OSError, ValueError, ValidationError):
                    age = now.timestamp() - directory.lstat().st_mtime
                    expired = age >= RETENTION_SECONDS
                if expired:
                    shutil.rmtree(directory, ignore_errors=True)
                    removed += 1
        except OSError:
            raise StorageFailure() from None
        return removed
