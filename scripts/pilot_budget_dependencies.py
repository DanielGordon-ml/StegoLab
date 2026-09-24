"""Derive independent CUDA runtime pins from the separate immutable lockfile."""

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any


def dependency_versions(directory: Path) -> dict[str, str]:
    """Walk runtime dependencies and selected extras in the Linux amd64 lock."""
    document = tomllib.loads((directory / "uv.lock").read_text(encoding="utf-8"))
    packages: dict[str, dict[str, Any]] = {
        package["name"]: package for package in document["package"]
    }
    pending: list[tuple[str, tuple[str, ...]]] = [
        (name, ())
        for name in ("numpy", "pillow", "pydantic", "pynacl", "reedsolo", "torch")
    ]
    visited: set[tuple[str, tuple[str, ...]]] = set()
    versions = {}
    while pending:
        name, extras = pending.pop()
        if (name, extras) in visited:
            continue
        visited.add((name, extras))
        package = packages[name]
        versions[name] = str(package["version"])
        dependencies = list(package.get("dependencies", []))
        for extra in extras:
            dependencies.extend(package.get("optional-dependencies", {})[extra])
        for dependency in dependencies:
            pending.append((dependency["name"], tuple(dependency.get("extra", []))))
    return dict(sorted(versions.items()))


def main() -> int:
    """Write or verify exact model-package dependencies without downloading wheels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    directory = Path(__file__).resolve().parents[1] / "infrastructure" / "cuda"
    content = (
        json.dumps(dependency_versions(directory), indent=2, sort_keys=True) + "\n"
    )
    target = directory / "runtime_dependencies.json"
    if arguments.check:
        return int(not target.exists() or target.read_text(encoding="utf-8") != content)
    target.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
