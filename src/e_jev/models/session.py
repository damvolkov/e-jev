"""models.session: what one `ejev` invocation knows before its command runs."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Session:
    """The run: its command path, where Jev answers, how the output is shaped."""

    command: str
    url: str
    key: str
    model: str
    timeout: float = 60.0
    json: bool = False
    verbose: bool = False
