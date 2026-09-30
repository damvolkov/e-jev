"""Unit-wide fixtures: a deterministic Reader, settings bound to a temp calibration path, the docs request."""

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import msgspec
import numpy as np
import pytest

from e_jev.adapters.ports import OpenReader
from e_jev.core.errors import RequestError
from e_jev.core.settings import Settings
from e_jev.models.reading import Reading
from e_jev.models.systemone import Json, SystemOneRequest

REFUSED = "readout refused"


class FakeReader:
    """Always prefers the first letter shown — a pure position bias — and costs one token per prompt word."""

    def __init__(self, up: bool = True) -> None:
        self.up = up

    async def logprobs(self, prompt: str, size: int) -> Reading:
        return Reading(scores=-np.arange(size, dtype=np.float64), input_tokens=len(prompt.split()), calls=1)

    async def extract(self, prompt: str, schema: dict[str, Any]) -> Json:
        return {"schema": sorted(schema)}

    async def health(self) -> bool:
        return self.up


class FailingReader(FakeReader):
    """A model server that is up but fails every readout."""

    async def logprobs(self, prompt: str, size: int) -> Reading:
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
