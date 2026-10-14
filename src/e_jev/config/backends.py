"""config.backends: the decision backends e-jev can route to, and the rows that order them per question kind."""

from enum import StrEnum, auto

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from e_jev.models.systemone import QuestionKind


class BackendKind(StrEnum):
    """How a backend is reached: today's logprob reader, a native `/v1/systemone` server, or Laya."""

    READING = auto()
    SYSTEMONE = auto()
    LAYA = auto()


class Language(StrEnum):
    """The language axis a state is routed by: the only one the contract must know, Spanish first."""

    ANY = auto()
    EN = auto()
    ES = auto()


class Backend(BaseModel, frozen=True):
    id: str
    kind: BackendKind
    url: str = ""
    api_key: str = "local"
    model: str = "jev-latest"
    languages: tuple[Language, ...] = (Language.ANY,)
    max_options: int = Field(255, ge=2, le=255)
    max_state_tokens: int = Field(262_144, ge=64)
    timeout: float = Field(120.0, gt=0)


READING = Backend(id="reading", kind=BackendKind.READING, url="", max_state_tokens=32_768)


class BackendsSettings(BaseSettings):
    ### JSON: JEV_BACKENDS='[{"id":"reading","kind":"reading"}, {"id":"clef","kind":"systemone","url":"http://localhost:45163","languages":["en"]}]'
    ### — the fleet; reading's `url` and `model` come from the vLLM settings, every other backend is reached at its own.
    backends: tuple[Backend, ...] = (READING,)
    ### JSON: JEV_ROWS='{"noul":["laya","reading"], "choice":["laya","clef","reading"]}' — preference per kind;
    ### a question descends its row on failure, so the last name in a row is the fleet's answer of last resort.
    rows: dict[QuestionKind, tuple[str, ...]] = {}
