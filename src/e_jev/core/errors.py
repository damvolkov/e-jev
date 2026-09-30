"""core.errors: the typed failure tree — one root, one agnostic base per action type."""


class JevError(Exception):
    """Root of every e-jev failure."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class RequestError(JevError):
    """An upstream call failed or answered out of contract."""


class ValidationError(JevError):
    """Input that parsed but cannot be honoured."""
