"""Hand-built zip and tar archives that exercise every extraction rule."""

import hashlib
import io
import os
import tarfile
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image

SYMLINK_ATTRIBUTES = 0o120777 << 16
ENCRYPTED_FLAG = 0x1
DIV2K_FOLDER = "DIV2K_valid_HR"
DIV2K_NAMES = ("0801.png", "0802.png", "0803.png")


def tiny_png(color: tuple[int, int, int] = (200, 30, 30)) -> bytes:
    """Return a real 4x4 PNG produced by Pillow."""
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buffer, format="PNG")
    return buffer.getvalue()


def tiny_jpeg() -> bytes:
    """Return a real 4x4 JPEG produced by Pillow."""
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (30, 200, 30)).save(buffer, format="JPEG")
    return buffer.getvalue()


def noisy_png(side: int = 64) -> bytes:
    """Return a PNG of random pixels that does not compress well."""
    buffer = io.BytesIO()
    image = Image.frombytes("RGB", (side, side), os.urandom(side * side * 3))
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@dataclass(frozen=True)
class ZipMember:
    """One zip entry with optional header lies applied after it is written."""

    name: str
    data: bytes
    symlink: bool = False
    encrypted: bool = False
    declared_size: int | None = None
    compress: int = zipfile.ZIP_STORED


def build_zip(members: list[ZipMember]) -> bytes:
    """Write members into a zip, then adjust central directory fields to lie."""
    buffer = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with zipfile.ZipFile(buffer, "w") as archive:
            for member in members:
                info = zipfile.ZipInfo(member.name)
                info.compress_type = member.compress
                if member.symlink:
                    info.external_attr = SYMLINK_ATTRIBUTES
                archive.writestr(info, member.data)
                if member.encrypted:
                    info.flag_bits |= ENCRYPTED_FLAG
                if member.declared_size is not None:
                    info.file_size = member.declared_size
    return buffer.getvalue()


@dataclass(frozen=True)
class TarMember:
    """One tar entry of any type, with a link target for link entries."""

    name: str
    data: bytes = b""
    entry_type: bytes = tarfile.REGTYPE
    linkname: str = ""


def build_tar(members: list[TarMember], *, compressed: bool = False) -> bytes:
    """Write members into a plain or gzip tar stream."""
    buffer = io.BytesIO()
    mode: Literal["w:gz", "w:"] = "w:gz" if compressed else "w:"
    with tarfile.open(fileobj=buffer, mode=mode) as archive:
        for member in members:
            info = tarfile.TarInfo(member.name)
            info.type = member.entry_type
            info.linkname = member.linkname
            if member.entry_type == tarfile.CHRTYPE:
                info.devmajor, info.devminor = 1, 3
            regular = member.entry_type == tarfile.REGTYPE
            info.size = len(member.data) if regular else 0
            archive.addfile(info, io.BytesIO(member.data) if regular else None)
    return buffer.getvalue()


def div2k_members() -> dict[str, bytes]:
    """Return DIV2K-shaped member paths with distinct real PNG bytes."""
    return {
        f"{DIV2K_FOLDER}/{name}": tiny_png((index * 60, 20, 200 - index * 50))
        for index, name in enumerate(DIV2K_NAMES)
    }


def div2k_zip() -> tuple[bytes, dict[str, str]]:
    """Build the normal DIV2K-shaped zip and the sha256 of every member."""
    members = div2k_members()
    archive = build_zip([ZipMember(name, data) for name, data in members.items()])
    digests = {name: hashlib.sha256(data).hexdigest() for name, data in members.items()}
    return archive, digests


def write_archive(directory: Path, name: str, data: bytes) -> Path:
    """Save archive bytes under a directory and return the path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(data)
    return path
