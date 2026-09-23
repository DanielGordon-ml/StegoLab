"""Validate explicit color declarations and prepare sRGB pixels."""

from io import BytesIO
from typing import Literal

from PIL import Image, ImageCms

from backend_service.image_validation import SourceHeader, image_failure

ColorPolicy = Literal["icc_to_srgb", "declared_srgb", "assumed_srgb"]


def resolve_color(
    image: Image.Image, header: SourceHeader
) -> tuple[ColorPolicy, ImageCms.ImageCmsProfile | None]:
    """Prefer an RGB profile, then an explicit sRGB tag, then an assumption."""
    profile_bytes = image.info.get("icc_profile")
    if profile_bytes is not None or b"iCCP" in header.png_chunks or header.has_profile:
        if not isinstance(profile_bytes, bytes) or not profile_bytes:
            raise image_failure("image_color")
        try:
            profile = ImageCms.ImageCmsProfile(BytesIO(profile_bytes))
            if profile.profile.xcolor_space.strip() != "RGB":
                raise image_failure("image_color")
            return "icc_to_srgb", profile
        except (OSError, ValueError, TypeError, ImageCms.PyCMSError):
            raise image_failure("image_color") from None
    if b"sRGB" in header.png_chunks:
        return "declared_srgb", None
    color_space = image.getexif().get_ifd(0x8769).get(0xA001)
    if color_space is not None:
        if color_space != 1:
            raise image_failure("image_color")
        return "declared_srgb", None
    return "assumed_srgb", None


def convert_color(
    image: Image.Image, profile: ImageCms.ImageCmsProfile | None
) -> Image.Image:
    """Convert RGB with fixed intent and preserve oriented alpha separately."""
    if profile is None:
        return image.copy()
    alpha = image.getchannel("A") if image.mode == "RGBA" else None
    try:
        converted = ImageCms.profileToProfile(
            image.convert("RGB"),
            profile,
            ImageCms.createProfile("sRGB"),
            renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
            outputMode="RGB",
            flags=ImageCms.Flags(0),
        )
        if converted is None:
            raise image_failure("image_color")
        if alpha is not None:
            converted.putalpha(alpha)
        return converted
    except (OSError, ValueError, TypeError, ImageCms.PyCMSError):
        raise image_failure("image_color") from None
