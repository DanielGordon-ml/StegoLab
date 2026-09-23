"""Check the repository rule that source files contain fewer than 300 lines."""

import subprocess
import sys
from pathlib import Path

SOURCE_SUFFIXES = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".css",
    ".sh",
    ".mjs",
    ".cjs",
    ".html",
}
SOURCE_NAMES = {"Dockerfile", "nginx.conf"}


def source_paths(repository: Path) -> list[Path]:
    """Find tracked and unignored source files without scanning installed packages."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    paths = {Path(name.decode()) for name in result.stdout.split(b"\0") if name}
    return sorted(
        repository / path
        for path in paths
        if path.suffix in SOURCE_SUFFIXES or path.name in SOURCE_NAMES
    )


def main() -> int:
    """Report oversized files and return a failure code when the rule is broken."""
    repository = Path(__file__).resolve().parents[1]
    failures = []
    paths = source_paths(repository)
    for path in paths:
        if not path.is_file():
            continue
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        if line_count >= 300:
            failures.append(f"{path.relative_to(repository)}: {line_count} lines")
    if failures:
        print("Source files must contain fewer than 300 lines:")
        print("\n".join(failures))
        return 1
    print(f"Checked {len(paths)} source files; all contain fewer than 300 lines.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
