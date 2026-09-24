"""Read-only checks for local worker output permissions and disk headroom."""

import os
import shutil
from pathlib import Path


def output_blockers(root: Path, operation: str, experiment: str) -> list[str]:
    """Check existing ancestors without creating folders or changing permissions."""
    folders = {"log": root / "logs"}
    if operation == "prepare_dataset":
        folders["dataset"] = root / "datasets"
    else:
        folders["budget"] = root / "state" / "cpu_proof"
    if operation == "train":
        folders["checkpoint"] = root / "checkpoints" / experiment
    if operation == "export":
        folders["model"] = root / "models"
    blockers: list[str] = []
    for label, destination in folders.items():
        ancestor = destination
        while not ancestor.exists() and ancestor.parent != ancestor:
            ancestor = ancestor.parent
        if not ancestor.is_dir() or not os.access(ancestor, os.W_OK | os.X_OK):
            blockers.append(
                f"The {label} output folder is not writable. "
                "Ask the server owner to check local folder permissions."
            )
            continue
        usage = shutil.disk_usage(ancestor)
        if usage.free <= max(10 * 1024**3, (usage.total + 9) // 10):
            blockers.append(
                "Storage is below the required free-space reserve. "
                "Free disk space before starting."
            )
    return list(dict.fromkeys(blockers))
