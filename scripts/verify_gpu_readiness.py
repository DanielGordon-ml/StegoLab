"""Prepare an honest report, or explicitly run a later existing-session GPU test."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    """Default to check-only; never launch an instance or start budget accounting."""
    from backend_service.dataset_serialization import read_bounded
    from backend_service.failures import ApplicationFailure
    from backend_service.pilot_readiness import verify_gpu_readiness
    from schemas.pilot_readiness import GpuReadinessRequest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    arguments = parser.parse_args()
    try:
        request = GpuReadinessRequest.model_validate_json(
            read_bounded(arguments.request, 64 * 1024)
        )
        report = verify_gpu_readiness(request, run=arguments.run)
    except ApplicationFailure as failure:
        print(json.dumps({"error": failure.code, "message": failure.message}))
        return 2
    except (OSError, ValueError):
        print(json.dumps({"error": "pilot_readiness_request_invalid"}))
        return 2
    print(report.model_dump_json(indent=2))
    if any(check.status == "failed" for check in report.checks):
        return 1
    return 0 if not arguments.run or report.readiness_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
