"""core.errors: the typed failure tree — one root, one agnostic base per action type."""

from typing import ClassVar


class JevError(Exception):
    """Root of every e-jev failure."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class RequestError(JevError):
    """An upstream call failed or answered out of contract."""


class ValidationError(JevError):
    """Input that parsed but cannot be honoured."""


class CliError(JevError):
    """Every expected `ejev` failure: the launcher prints the message and exits with the class's code."""

    exit_code: ClassVar[int] = 1
