"""tests.kit: fleet doubles and builders shared by every tier — fixtures stay in conftest, code lives here."""

from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from json import dumps
from typing import Any
from zlib import crc32

import numpy as np
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Response

from e_jev.adapters.decider.reading import ReadingDecider
from e_jev.adapters.ports import Decider, OpenFleet, Reader
from e_jev.config.backends import READING, Backend, BackendKind, Language
from e_jev.core.errors import OverloadedError, RateLimitedError, RequestError
from e_jev.logic.dispatch import option_count
from e_jev.models.decision import Decision
from e_jev.models.reading import Scores
from e_jev.models.systemone import Json, Question

REFUSED = "readout refused"


class FakeReader:
    """Always prefers the lowest candidate token — the first label shown, or stopping — a pure position bias.

    Labels are their characters' code points; the stop token is 0; a prompt is one token per word, by content."""

    def __init__(self, up: bool = True) -> None:
        self.up = up

    @property
    def stop(self) -> int:
        return 0

    def label(self, text: str) -> tuple[int, ...]:
        return tuple(map(ord, text))

    async def encode(self, prompt: str) -> tuple[int, ...]:
        return tuple(crc32(word.encode()) for word in prompt.split())

    async def logprobs(self, tokens: Sequence[int], candidates: Sequence[int]) -> Scores:
        return -np.arange(len(candidates), dtype=np.float64)

    async def extract(self, prompt: str, schema: dict[str, Any]) -> Json:
        return {"schema": sorted(schema)}

    async def health(self) -> bool:
        return self.up


class FailingReader(FakeReader):
    """A model server that is up but fails every readout."""

    async def logprobs(self, tokens: Sequence[int], candidates: Sequence[int]) -> Scores:
        raise RequestError(REFUSED)


class RateLimitedReader(FailingReader):
    """A model server that answers 429: too many requests, with a retry hint."""

    async def logprobs(self, tokens: Sequence[int], candidates: Sequence[int]) -> Scores:
        raise RateLimitedError(REFUSED, retry_after=2.0)


class OverloadedReader(FailingReader):
    """A model server at capacity: it sheds every readout."""

    async def logprobs(self, tokens: Sequence[int], candidates: Sequence[int]) -> Scores:
        raise OverloadedError(REFUSED)


class FakeDecider:
    """A routing double: one backend answering with a fixed descending distribution, recording what it was asked."""

    def __init__(self, backend: str, *, up: bool = True, failing: Callable[[str], Exception] | None = None) -> None:
        self._backend = backend
        self.up = up
        self._failing = failing
        self.seen: list[tuple[str, ...]] = []

    @property
    def backend(self) -> str:
        return self._backend

    async def decide(self, state: str, questions: Mapping[str, Question]) -> Decision:
        self.seen.append(tuple(questions))
        match self._failing:
            case None:
                scores = {name: np.log(np.linspace(0.9, 0.1, option_count(question))) for name, question in questions.items()}
                return Decision(scores=scores, input_tokens=10, output_tokens=1)
            case error:
                raise error(REFUSED)

    async def health(self) -> bool:
        return self.up


def _json(payload: dict[str, Any], status: int) -> Response:
    return Response(dumps(payload), status=status, content_type="application/json")


class FakeVllm:
    """The tokenize dialect double: down, every request 404s like vLLM's missing model does; up, the fixed
    probe renders stop=4 through the chat template and every label tokenizes."""

    def __init__(self, server: HTTPServer) -> None:
        self._server = server
        self.up = False
        self.attempts = 0

    def url_for(self, path: str) -> str:
        return self._server.url_for(path)

    def respond(self, request: Any) -> Response:
        self.attempts += 1
        match (request.path, request.method, self.up):
            case ("/health", "GET", _):
                return _json({"status": "ok"}, 200)
            case ("/tokenize", "POST", False):
                return _json(
                    {"error": {"message": f"The model `{request.json['model']}` does not exist.", "type": "NotFoundError", "code": 404}},
                    404,
                )
            case ("/tokenize", "POST", True):
                return _json(
                    {"tokens": [1, 2, 3, 4, 5, 6, 7] if "messages" in request.json else [1, 2, 3], "count": 7, "max_model_len": 65536}, 200
                )
            case _:
                return _json({"error": {"message": f"no route {request.path}"}}, 404)


def backend(backend_id: str, kind: BackendKind, *, languages: tuple[Language, ...] = (Language.ANY,), **limits: Any) -> Backend:
    return Backend(id=backend_id, kind=kind, url=f"http://{backend_id}", languages=languages, **limits)


def reading_spec(model: str = "m") -> Backend:
    return READING.model_copy(update={"model": model})


def fleet_of(reader: Reader, *, permutations: int = 1, concurrency: int = 4) -> dict[str, Decider]:
    return {"reading": ReadingDecider(reader, reading_spec(), permutations=permutations, concurrency=concurrency)}


@asynccontextmanager
async def _as_fleet(deciders: Mapping[str, Decider]) -> AsyncIterator[Mapping[str, Decider]]:
    yield deciders


def open_fleet(deciders: Mapping[str, Decider]) -> OpenFleet:
    return lambda _: _as_fleet(deciders)
