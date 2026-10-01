"""models.systemone: the System One wire contract, mirroring TypeSafe's public API (docs.typesafe.ai/api)."""

from enum import StrEnum, auto
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
    criteria: Annotated[dict[str, Json | None], Meta(min_length=2, max_length=255)]
    instructions: Json | None = None


class Score(Struct, frozen=True, tag="score", tag_field="type"):
    criteria: Annotated[list[Json], Meta(min_length=2, max_length=10)]
    instructions: Json | None = None


type Question = Noul | Choice | Score


class QuestionKind(StrEnum):
    NOUL = auto()
    CHOICE = auto()
    SCORE = auto()

    @classmethod
    def of(cls, question: Question) -> "QuestionKind":
        return cls(question.__struct_config__.tag)


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
