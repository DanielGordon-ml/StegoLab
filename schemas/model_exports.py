"""Strict records for independent experimental CPU model packages."""

from typing import Literal

from pydantic import Field, field_validator, model_validator

from schemas.base import StrictRecord


class ExportMetadata(StrictRecord):
    """Bind deployment packages to the exact trained model pair."""

    compatibility_identifier: str = Field(pattern=r"^dense_v1_[0-9a-f]{64}$")
    source_identifier: str = Field(min_length=1, max_length=256)


class ExportFile(StrictRecord):
    """Describe one bounded, checksummed package file."""

    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=1, le=16 * 1024**2)


class ModelExportManifest(ExportMetadata):
    """Freeze the tensor interface, protocol, and independent dependencies."""

    format_version: Literal[1, 2] = 1
    role: Literal["encoder", "decoder"]
    architecture_identifier: Literal["dense_residual_v1"] = "dense_residual_v1"
    status: Literal["experimental"] = "experimental"
    graph_filename: Literal["model.pt2"] = "model.pt2"
    tensor_dtype: Literal["float32"] = "float32"
    tensor_layout: Literal["batch_channels_height_width"] = (
        "batch_channels_height_width"
    )
    batch_size: Literal[1] = 1
    minimum_side: Literal[512] = 512
    maximum_side: Literal[1024] = 1024
    protocol_version: Literal[1] = 1
    profile_identifier: Literal["test_only_v1"] = "test_only_v1"
    image_policy: Literal["rgb_zero_to_one_round_255"] = "rgb_zero_to_one_round_255"
    capacity_rule: Literal["min(1024, width * height // 1024)"] = (
        "min(1024, width * height // 1024)"
    )
    python_version: Literal["3.12"] = "3.12"
    dependencies: dict[str, str] = Field(min_length=1, max_length=32)
    runtime_dependencies: dict[str, dict[str, str]] = Field(default_factory=dict)
    producer_environment: dict[str, str] = Field(min_length=1, max_length=8)
    files: dict[str, ExportFile] = Field(min_length=1, max_length=64)

    @field_validator("format_version", mode="before")
    @classmethod
    def require_integer_format(cls, value: object) -> object:
        """Reject booleans and decimal lookalikes at the package format boundary."""
        if type(value) is not int:
            raise ValueError("The package format version requires a whole number.")
        return value

    @model_validator(mode="after")
    def validate_runtime_dependencies(self) -> "ModelExportManifest":
        """Require explicit tested CPU pins and separately recorded CUDA pins."""
        if self.format_version == 1:
            if self.runtime_dependencies:
                raise ValueError("Version one packages support only CPU dependencies.")
        elif (
            set(self.runtime_dependencies) != {"cpu", "cuda"}
            or self.runtime_dependencies["cpu"] != self.dependencies
            or any(
                not 1 <= len(pins) <= 128 for pins in self.runtime_dependencies.values()
            )
            or self.runtime_dependencies["cuda"].get("torch") != "2.14.0+cu126"
        ):
            raise ValueError("Version two packages require CPU and CUDA runtime pins.")
        return self


class ModelExportSummary(ExportMetadata):
    """Return public package locations without training or secret inputs."""

    directory: str
    encoder_directory: str
    decoder_directory: str
    status: Literal["experimental"] = "experimental"


class ExportVerification(StrictRecord):
    """Record clean-process tensor agreement for an independent pair."""

    compatibility_identifier: str = Field(pattern=r"^dense_v1_[0-9a-f]{64}$")
    cases_checked: int = Field(ge=1)
    maximum_absolute_difference: float = Field(ge=0, allow_inf_nan=False)
    relative_tolerance: float = Field(default=0.00001, ge=0.00001, le=0.00001)
    absolute_tolerance: float = Field(default=0.00001, ge=0.00001, le=0.00001)
    status: Literal["passed"] = "passed"
    device: Literal["cpu", "cuda"] = "cpu"
