"""Application failures with messages safe for clients and logs."""


class ApplicationFailure(Exception):
    """Carry a fixed safe explanation without private exception details."""

    def __init__(self, code: str, message: str, status_code: int = 503) -> None:
        """Set the response code and public explanation."""
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class StorageFailure(ApplicationFailure):
    """Report unavailable or invalid saved state without overwriting it."""

    def __init__(self) -> None:
        """Provide a recovery message that contains no storage contents."""
        super().__init__(
            "storage_unavailable",
            "Saved data could not be read or written. Check storage access and "
            "free space, then retry. Existing saved data has been kept.",
        )
