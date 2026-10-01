"""Unit-wide fixtures: a deterministic Reader, settings bound to a temp calibration path, the docs request."""

from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from zlib import crc32

import msgspec
import numpy as np
import pytest

from e_jev.adapters.ports import OpenReader
from e_jev.core.errors import RequestError
from e_jev.core.settings import Settings
from e_jev.models.reading import Scores
from e_jev.models.systemone import Json, SystemOneRequest

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


@pytest.fixture
def reader() -> FakeReader:
    return FakeReader()


@pytest.fixture
def open_reader(reader: FakeReader) -> OpenReader:
    return lambda _: nullcontext(reader)


@pytest.fixture
def open_reader_down() -> OpenReader:
    return lambda _: nullcontext(FakeReader(up=False))


@pytest.fixture
def open_reader_failing() -> OpenReader:
    return lambda _: nullcontext(FailingReader())


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(calibration=tmp_path / "calibration.json", model="m", env="dev")


@pytest.fixture
def request_systemone(request_bytes: bytes) -> SystemOneRequest:
    return msgspec.json.decode(request_bytes, type=SystemOneRequest)
