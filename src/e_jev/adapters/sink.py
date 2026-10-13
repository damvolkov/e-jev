"""adapters.sink: answered decisions as JSONL — the corpus the distillation loop trains fast heads from.

One line per answered question: state, question, the raw scores, the backend that produced them and the
request id. Append-only, line-buffered, flushed every line: a crash loses nothing already written. This is
the teacher's syllabus (route `reading`'s distribution), so the same state can carry the slow path's exact
probabilities while the fast paths serve the live traffic."""

import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Self

import msgspec
import structlog

from e_jev.core.settings import Settings
from e_jev.models.systemone import Question

log = structlog.get_logger()


class JsonlSink:
    __slots__ = ("_path", "_write")

    def __init__(self, path: Path) -> None:
        self._path = path
        self._write = path.open("ab", buffering=0)

    @classmethod
    @asynccontextmanager
    async def open(cls, settings: Settings) -> AsyncIterator[Self | None]:
        """The sink at `JEV_DECISION_LOG`, or nothing if the path is unset — logging is opt-in."""
        match settings.decision_log:
            case None:
                yield None
            case path:
                path.parent.mkdir(parents=True, exist_ok=True)
                sink = cls(path)
                log.info("decision_log_open", path=str(path))
                try:
                    yield sink
                finally:
                    sink.close()

    def record(self, state: str, question: Question, scores: Sequence[float], backend: str) -> None:
        line = {"ts": time.time(), "backend": backend, "state": state, "question": question, "scores": list(scores)}
        self._write.write(msgspec.json.encode(line) + b"\n")

    def close(self) -> None:
        self._write.close()
