"""Server-side Hugging Face token lookup; the token never leaves the backend."""

import os
from pathlib import Path

MAXIMUM_TOKEN_BYTES = 4096


def read_hugging_face_token() -> str | None:
    """Read the token from HF_TOKEN_FILE, then HF_TOKEN; blanks count as absent."""
    file_name = os.environ.get("HF_TOKEN_FILE", "")
    if file_name:
        try:
            with Path(file_name).open("r", encoding="utf-8") as source:
                value = source.read(MAXIMUM_TOKEN_BYTES + 1).strip()
        except (OSError, UnicodeDecodeError):
            value = ""
        if value and len(value) <= MAXIMUM_TOKEN_BYTES:
            return value
    value = os.environ.get("HF_TOKEN", "").strip()
    if value and len(value) <= MAXIMUM_TOKEN_BYTES:
        return value
    return None


def hugging_face_token_configured() -> bool:
    """Report only whether a token exists, never its value."""
    return read_hugging_face_token() is not None
