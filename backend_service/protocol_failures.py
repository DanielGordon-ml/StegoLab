"""Fixed safe failures for protocol validation and message recovery."""

from backend_service.failures import ApplicationFailure

RECOVERY_MESSAGE = (
    "No valid hidden message could be recovered. Check the password, model, and image."
)


def recovery_failure() -> ApplicationFailure:
    """Hide whether missing data, damage, or authentication caused failure."""
    return ApplicationFailure("message_recovery_failed", RECOVERY_MESSAGE, 422)


def invalid_protocol_input() -> ApplicationFailure:
    """Reject an unsupported context or layout without including its values."""
    return ApplicationFailure(
        "invalid_protocol_input",
        "Use supported image dimensions and the matching protocol profile.",
        422,
    )


def protocol_resource_failure() -> ApplicationFailure:
    """Handle allocation failures without exposing library exceptions."""
    return ApplicationFailure(
        "protocol_resources_unavailable",
        "The message could not be processed. Free memory and retry.",
    )
