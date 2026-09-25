"""Fixed failures and bounded file helpers shared by inference storage."""

import os
import stat
from pathlib import Path

from backend_service.failures import ApplicationFailure
from schemas.images import ImageSummary


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


def directory_bytes(directory: Path) -> int:
    """Sum the regular files kept directly inside each child directory."""
    total = 0
    for child in directory.iterdir():
        try:
            if child.is_symlink() or not child.is_dir():
                continue
            for entry in child.iterdir():
                information = entry.lstat()
                if stat.S_ISREG(information.st_mode):
                    total += information.st_size
        except FileNotFoundError:
            # A concurrent sweep or scratch cleanup removed it; nothing to count.
            continue
    return total
