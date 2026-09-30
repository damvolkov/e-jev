"""Wire and domain types. The System One shapes mirror TypeSafe's public API (docs.typesafe.ai/api)."""

from typing import Annotated, Any

from msgspec import Meta, Struct

type Json = str | dict[str, Any] | list[Any]


class NoulCriteria(Struct, frozen=True):
    true: Json | None = None
    false: Json | None = None


class Noul(Struct, frozen=True, tag="noul", tag_field="type"):
    instructions: Json | None = None
    criteria: NoulCriteria | None = None


class Choice(Struct, frozen=True, tag="choice", tag_field="type"):
    ### Jev accepts 255 options; single-token letters cap this readout at 26.
    criteria: Annotated[dict[str, Json | None], Meta(min_length=2, max_length=26)]
    instructions: Json | None = None


class Score(Struct, frozen=True, tag="score", tag_field="type"):
    criteria: Annotated[list[Json], Meta(min_length=2, max_length=10)]
    instructions: Json | None = None


type Question = Noul | Choice | Score


class SystemOneRequest(Struct, frozen=True):
    state: Json
    questions: Annotated[dict[str, Question], Meta(min_length=1)]
    model: str = "jev-latest"


class NoulAnswer(Struct, frozen=True, tag="noul", tag_field="type"):
    noul: float


class ChoiceAnswer(Struct, frozen=True, tag="choice", tag_field="type"):
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(Struct, frozen=True, tag="score", tag_field="type"):
    score: float
    legend: dict[str, Json]
    probabilities: dict[str, float]
    confidence: float


type Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


class Usage(Struct, frozen=True):
    input_tokens: int
    output_tokens: int


class SystemOneResponse(Struct, frozen=True):
    model: str
    answers: dict[str, Answer]
    usage: Usage


class ModelMetadata(Struct, frozen=True):
    name: str
    description: str
    release_date: str


class ModelList(Struct, frozen=True):
    models: tuple[ModelMetadata, ...]


class ExtractRequest(Struct, frozen=True):
    state: Json
    instructions: Json
    schema: dict[str, Any]


class Labeled(Struct, frozen=True):
    """One calibration example: the choice key, the score level index, or "true"/"false" for a noul."""

    state: Json
    question: Question
    label: str


class Calibration(Struct, frozen=True):
    model: str
    permutations: int
    temperature: float
    ece_before: float
    ece_after: float
    samples: int
