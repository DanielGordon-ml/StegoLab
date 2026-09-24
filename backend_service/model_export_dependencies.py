"""Exact common inference dependencies for portable CPU graph packages."""

import importlib.metadata

COMMON_DEPENDENCIES = (
    "torch",
    "numpy",
    "Pillow",
    "PyNaCl",
    "reedsolo",
    "pydantic",
    "cffi",
    "pycparser",
    "pydantic_core",
    "typing_extensions",
    "typing-inspection",
    "annotated-types",
    "filelock",
    "sympy",
    "mpmath",
    "networkx",
    "Jinja2",
    "MarkupSafe",
    "fsspec",
)


def cpu_dependencies() -> dict[str, str]:
    """Pin shared inference packages while allowing native CPU Torch suffixes."""
    versions = {name: importlib.metadata.version(name) for name in COMMON_DEPENDENCIES}
    versions["torch"] = versions["torch"].split("+", 1)[0]
    return versions
