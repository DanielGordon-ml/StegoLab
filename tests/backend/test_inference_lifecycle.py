"""Secret lifecycle, scheduler rules, and bounds for encode and decode jobs."""

import fcntl
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from inference_fixtures import (
    image_bytes,
)
from test_inference_jobs import PASSWORD, assert_secrets_absent

from backend_service.failures import ApplicationFailure
from backend_service.inference_files import InferenceFileStore
from backend_service.inference_jobs import InferenceServices, submit_inference
from backend_service.inference_secrets import DecodedTextStore, InferenceSecretStore
from backend_service.model_export_io import export_failure
from backend_service.model_fixture_channel import FIXTURE_PACKAGE_NAME
from backend_service.model_installation import InstalledModelStore
from backend_service.workflow_supervision import next_job, report_failure
from backend_service.workspace_catalog import WorkspaceCatalog
from backend_service.workspace_jobs import WorkspaceJobService
from schemas.configuration import ConfigurationProfile
from schemas.inference_jobs import (
    DecodingJobRequest,
    EncodingJobRequest,
)
from schemas.jobs import JobSnapshot
from schemas.models import ModelInstallRequest
from schemas.workflows import WorkflowActionRequest, WorkflowRequest


def direct_service(
    workspace: Path, state: Path
) -> tuple[WorkspaceJobService, InferenceServices, str]:
    """Build the job supervisor with inference stores but without a scheduler."""
    catalog = WorkspaceCatalog(workspace, state)
    service = WorkspaceJobService(catalog, state)
    installed = InstalledModelStore(state, catalog.registry.roots)
    installed.install(
        ModelInstallRequest(client_request_identifier="install", export_reference="x"),
        lambda: (workspace / "models" / FIXTURE_PACKAGE_NAME, catalog.root),
    )
    service.inference = InferenceServices(
        files=InferenceFileStore(state),
        installed=installed,
        secrets=InferenceSecretStore(),
        texts=DecodedTextStore(),
    )
    return service, service.inference, FIXTURE_PACKAGE_NAME


def test_cancel_drops_secrets_restart_loses_inputs_and_proof_lock_is_ignored(
    fixture_workspace: Path, tmp_path: Path
) -> None:
    """Keep secrets only while a job waits, and never block inference on the proof."""
    state = tmp_path / "state"
    service, inference, model = direct_service(fixture_workspace, state)
    cover = inference.files.store("cover", image_bytes()).image_reference
    encoded = inference.files.store("encoded", image_bytes()).image_reference
    first = submit_inference(
        service,
        EncodingJobRequest(
            client_request_identifier="encode",
            image_reference=cover,
            model_identifier=model,
            message="hello",
            password=PASSWORD,
        ),
    )
    second = submit_inference(
        service,
        DecodingJobRequest(
            client_request_identifier="decode",
            image_reference=encoded,
            model_identifier=model,
            password=PASSWORD,
        ),
    )
    assert len(inference.secrets) == 2
    cancelled = service.action(
        first.job_identifier,
        WorkflowActionRequest(client_request_identifier="cancel", action="cancel"),
    )
    assert cancelled.status == "cancelled"
    assert len(inference.secrets) == 1
    earlier = datetime.now(UTC) - timedelta(minutes=5)
    training = JobSnapshot(
        job_identifier="job_training_first",
        status="queued",
        phase="queued",
        configuration=ConfigurationProfile(),
        available_actions=["cancel"],
        created_at=earlier,
        updated_at=earlier,
        operation="train",
        experiment_identifier="cpu_experiment",
    )
    service.store.save(
        training,
        request=WorkflowRequest(client_request_identifier="train", operation="train"),
    )
    proof = fixture_workspace / "state" / "cpu_proof"
    proof.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(proof / ".proof.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = next_job(service)
    finally:
        os.close(descriptor)
    assert started is not None and started.job_identifier == second.job_identifier
    assert service.get_job("job_training_first").status == "queued"
    service.closing.set()
    report_failure(service, second.job_identifier, export_failure())
    stopped = service.get_job(second.job_identifier)
    assert stopped.status == "interrupted"
    assert stopped.error is not None and stopped.error["code"] == "inputs_lost"
    service.closing.clear()
    reloaded, _, _ = direct_service(fixture_workspace, state)
    reloaded.start()
    try:
        interrupted = reloaded.get_job(second.job_identifier)
        assert interrupted.status == "interrupted"
        assert interrupted.error is not None
        assert interrupted.error["code"] == "inputs_lost"
        assert reloaded.get_job(first.job_identifier).error is None
        assert reloaded.get_job("job_training_first").status == "interrupted"
    finally:
        reloaded.close()
    assert_secrets_absent(state)


def test_refused_second_supervisor_leaves_the_owner_queue_alone(
    fixture_workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second application on the same state must not interrupt the owner's jobs."""
    monkeypatch.setattr(
        "backend_service.workflow_supervision.next_job", lambda service: None
    )
    state = tmp_path / "state"
    owner, inference, model = direct_service(fixture_workspace, state)
    encoded = inference.files.store("encoded", image_bytes()).image_reference
    owner.start()
    try:
        job = submit_inference(
            owner,
            DecodingJobRequest(
                client_request_identifier="decode",
                image_reference=encoded,
                model_identifier=model,
                password=PASSWORD,
            ),
        )
        second, _, _ = direct_service(fixture_workspace, state)
        with pytest.raises(ApplicationFailure, match="Another application"):
            second.start()
        second.close()
        assert owner.get_job(job.job_identifier).status == "queued"
        assert len(inference.secrets) == 1
    finally:
        owner.close()
    assert owner.get_job(job.job_identifier).status == "interrupted"


def test_queue_bound_and_decoded_text_expiry(
    fixture_workspace: Path, tmp_path: Path
) -> None:
    """Refuse a ninth waiting job and forget recovered text after its lifetime."""
    service, inference, model = direct_service(fixture_workspace, tmp_path / "state")
    encoded = inference.files.store("encoded", image_bytes()).image_reference
    for index in range(8):
        submit_inference(
            service,
            DecodingJobRequest(
                client_request_identifier=f"decode-{index}",
                image_reference=encoded,
                model_identifier=model,
                password="p",
            ),
        )
    with pytest.raises(Exception, match="already waiting"):
        submit_inference(
            service,
            DecodingJobRequest(
                client_request_identifier="decode-9",
                image_reference=encoded,
                model_identifier=model,
                password="p",
            ),
        )
    moment = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)
    texts = DecodedTextStore(lifetime_seconds=300, clock=lambda: moment)
    for index in range(12):
        texts.put(f"job_{index}", f"text {index}")
    first = texts.get("job_0")
    assert first is not None and first.expires_at == moment + timedelta(seconds=300)
    moment += timedelta(seconds=300)
    assert texts.get("job_0") is None and texts.get("job_11") is None
    assert texts.sweep() == 0
