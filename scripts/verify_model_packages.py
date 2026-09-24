"""Measure independent native or Linux CPU text recovery inside the proof ledger."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import torch

from backend_service.model_data import load_proof_data
from backend_service.model_export_recovery import verify_package_recovery
from backend_service.proof_runtime import run_directory, write_record
from backend_service.training_budget import ProofBudget
from schemas.export_recovery import ExportRecoveryReport


def container_report(
    image: str, package: Path, dataset: Path, remaining_seconds: float
) -> ExportRecoveryReport:
    """Use a network-disabled non-root Linux runtime with read-only inputs."""
    program = (
        "import json,sys,time,torch; from pathlib import Path; "
        "from backend_service.model_data import load_proof_data; "
        "from backend_service.model_export_recovery import verify_package_recovery; "
        "torch.set_num_threads(4); deadline=time.monotonic()+float(sys.argv[1]); "
        "data=load_proof_data(Path('/proof/data')/sys.argv[2],deadline=deadline); "
        "report=verify_package_recovery(Path('/proof/package'),data,"
        "deadline=deadline); "
        "print(report.model_dump_json())"
    )
    name = f"stegolab-package-proof-{uuid4().hex}"
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--memory",
        "10g",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--tmpfs",
        "/tmp:size=64m,mode=1777",
        "--mount",
        f"type=bind,source={package},target=/proof/package,readonly",
        "--mount",
        f"type=bind,source={dataset.parent},target=/proof/data,readonly",
        image,
        "python",
        "-c",
        program,
        str(remaining_seconds),
        dataset.name,
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, check=True, timeout=remaining_seconds
        )
    finally:
        subprocess.run(
            ["docker", "rm", "--force", name],
            capture_output=True,
            check=False,
            timeout=15,
        )
    return ExportRecoveryReport.model_validate_json(result.stdout)


def main() -> None:
    """Save complete or incomplete package evidence without extending time limits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument("--container-image")
    arguments = parser.parse_args()
    root = arguments.output_root.resolve()
    package, dataset = arguments.package.resolve(), arguments.dataset.resolve()
    with ProofBudget(
        root / "state" / "cpu_proof", arguments.experiment, resume=True
    ) as budget:
        if arguments.container_image:
            report = container_report(
                arguments.container_image, package, dataset, budget.remaining_seconds
            )
            environment = "linux_cpu"
        else:
            torch.set_num_threads(4)
            deadline = time.monotonic() + budget.remaining_seconds
            data = load_proof_data(dataset, deadline=deadline)
            report = verify_package_recovery(package, data, deadline=deadline)
            environment = "native_cpu"
        directory = run_directory(root, arguments.experiment)
        destination = directory / f"{environment}_package_recovery.json"
        write_record(destination, report)
        print(
            json.dumps(
                {
                    "report": str(destination.relative_to(root)),
                    "environment": environment,
                    "completed": report.completed,
                    "exact_recovery_count": report.exact_recovery_count,
                    "recovery_gate_passed": report.recovery_gate_passed,
                },
                indent=2,
            )
        )
        if not report.recovery_gate_passed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
