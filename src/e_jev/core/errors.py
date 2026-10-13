"""core.errors: the typed failure tree — one root, one agnostic base per action type."""

from collections.abc import Mapping
from enum import StrEnum, auto
from typing import ClassVar, Final


class JevError(Exception):
    """Root of every e-jev failure."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class RequestError(JevError):
    """An upstream call failed or answered out of contract."""


class CapabilityError(JevError):
    """No backend in the fleet serves this capability: 501, and it will not change by asking again."""


class Capacity(StrEnum):
    """The statuses TypeSafe's SDKs retry: 429 the client sent too fast, 529 the server is at capacity."""

    RATE_LIMITED = auto()
    OVERLOADED = auto()


STATUSES: Final[Mapping[Capacity, int]] = {Capacity.RATE_LIMITED: 429, Capacity.OVERLOADED: 529}


class RetryableError(RequestError):
    """The model server is at capacity. TypeSafe's SDKs retry exactly these statuses — 429 and 529 —
    so a 502 here would discard a decision that only needed a moment."""

    capacity: ClassVar[Capacity]

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after

    @staticmethod
    def _open_leaves(error: BaseException) -> list["RetryableError"]:
        """The retryable leaves of a possibly nested group of parallel readout failures."""
        match error:
            case BaseExceptionGroup(exceptions=inner):
                return [leaf for nested in inner for leaf in RetryableError._open_leaves(nested)]
            case RetryableError() as retryable:
                return [retryable]
            case _:
                return []

    @classmethod
    def from_group(cls, failures: BaseException) -> "RetryableError":
        """One capacity failure for parallel readouts: the rate limit wins, keeping its retry hint."""
        leaves = cls._open_leaves(failures)
        message = "; ".join(sorted({leaf.message for leaf in leaves}))
        limited = next((leaf for leaf in leaves if type(leaf) is RateLimitedError), None)
        ### One binary choice: a row that saw a 429 anywhere is answered with its 429, else every leaf is a 529.
        return RateLimitedError(message, limited.retry_after) if limited else OverloadedError(message)


class RateLimitedError(RetryableError):
    """The client is over its rate: 429, with Retry-After when the upstream hints one."""

    capacity: ClassVar[Capacity] = Capacity.RATE_LIMITED


class OverloadedError(RetryableError):
    """The server is at capacity or went silent: 529, Jev's overload status."""

    capacity: ClassVar[Capacity] = Capacity.OVERLOADED


class ValidationError(JevError):
    """Input that parsed but cannot be honoured."""


class UnroutableError(JevError):
    """No backend in the fleet may answer this question: the state is beyond every budget, or the row names none."""

    def __init__(self, kind: object, name: str, row: list[str]) -> None:
        super().__init__(f"question {name!r} ({kind}) is unroutable: row {row or 'empty'} holds no eligible backend")
        self.kind = kind
        self.name = name


class CliError(JevError):
    """Every expected `ejev` failure: the launcher prints the message and exits with the class's code."""

    exit_code: ClassVar[int] = 1
