"""cli.services.decide: one System One request built from terminal input — no terminal, no rendering here."""

from enum import StrEnum, auto
from pathlib import Path
from typing import Any, Final

import msgspec
from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, Score
from typesafe_sdk._core.response_types import ListModelsResponse, SystemOneResponse

from e_jev.core.errors import CliError
from e_jev.core.telemetry import inject_headers

type Question = Noul | Choice | Score


class DecideKind(StrEnum):
    CHOICE = auto()
    SCORE = auto()


CHOICE_OPTIONS: Final = range(2, 256)
SCORE_LEVELS: Final = range(2, 11)


class DecideInputError(CliError):
    """What the terminal gave cannot become a System One request."""

    exit_code = 2


class DecideSizeError(DecideInputError):
    """A choice or a score with a count of options Jev does not accept."""

    def __init__(self, kind: DecideKind, allowed: range, got: int) -> None:
        super().__init__(f"a {kind} needs {allowed.start} to {allowed.stop - 1}, got {got}")
        self.kind = kind
        self.got = got


class DecideFileError(DecideInputError):
    """A state or request file that cannot be read, or is not what the command needs."""

    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(f"{path}: {reason}")
        self.path = path


class DecideService:
    """Builds questions from flags, reads states from text or files, and asks the endpoint once per call."""

    __slots__ = ("_client", "_model")

    def __init__(self, client: AsyncTypeSafeClient, model: str) -> None:
        self._client = client
        self._model = model

    ##### PRIVATE #####

    @staticmethod
    def _state_file(path: Path) -> Any:
        """A `.json` file is structured state, anything else plain text."""
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise DecideFileError(path, f"cannot read state file ({error.strerror})") from error
        return msgspec.json.decode(raw) if path.suffix == ".json" else raw.decode()

    ############################################################

    ##### PUBLIC #####

    @classmethod
    def state(cls, raw: str) -> Any:
        """The state as given, or the contents of `@path`, curl style."""
        return cls._state_file(Path(raw[1:])) if raw.startswith("@") else raw

    @staticmethod
    def noul(question: str, true: str = "", false: str = "") -> Noul:
        criteria = {key: value for key, value in (("true", true), ("false", false)) if value}
        return Noul(instructions=question, criteria=criteria or None)

    @staticmethod
    def choice(question: str, options: tuple[str, ...]) -> Choice:
        """Options as `name` or `name=description`; Jev takes 2 to 255."""
        criteria = {name: description or None for name, _, description in (option.partition("=") for option in options)}
        match len(criteria):
            case size if size in CHOICE_OPTIONS:
                return Choice(instructions=question, criteria=criteria)
            case size:
                raise DecideSizeError(DecideKind.CHOICE, CHOICE_OPTIONS, size)

    @staticmethod
    def score(question: str, levels: tuple[str, ...]) -> Score:
        """Levels lowest first; Jev takes 2 to 10."""
        match len(levels):
            case size if size in SCORE_LEVELS:
                return Score(instructions=question, criteria=list(levels))
            case size:
                raise DecideSizeError(DecideKind.SCORE, SCORE_LEVELS, size)

    @classmethod
    def request(cls, path: Path) -> tuple[Any, dict[str, Any], str | None]:
        """A full TypeSafe request body from a file: state, questions and, optionally, the model."""
        match cls._state_file(path):
            case {"state": state, "questions": dict() as questions, **rest}:
                return state, questions, rest.get("model")
            case _:
                raise DecideFileError(path, "not a System One request: it needs `state` and a `questions` map")

    async def ask(self, state: Any, questions: dict[str, Question] | dict[str, Any], model: str | None = None) -> SystemOneResponse:
        return await self._client.system_one(state, questions, model=model or self._model, extra_headers=inject_headers())

    async def models(self) -> ListModelsResponse:
        return await self._client.models.list()
