"""Unit-wide fixtures: the deterministic fleet and settings bound to a temp calibration path."""

from pathlib import Path

import msgspec
import pytest
from tests import kit

from e_jev.adapters.ports import Decider, OpenFleet
from e_jev.core.settings import Settings
from e_jev.models.systemone import SystemOneRequest


@pytest.fixture
def reader() -> kit.FakeReader:
    return kit.FakeReader()


@pytest.fixture
def deciders(reader: kit.FakeReader) -> dict[str, Decider]:
    return kit.fleet_of(reader)


@pytest.fixture
def open_reader_fleet(reader: kit.FakeReader) -> OpenFleet:
    return kit.open_fleet(kit.fleet_of(reader))


@pytest.fixture
def open_fleet_down() -> OpenFleet:
    return kit.open_fleet(kit.fleet_of(kit.FakeReader(up=False)))


@pytest.fixture
def open_fleet_failing() -> OpenFleet:
    return kit.open_fleet(kit.fleet_of(kit.FailingReader()))


@pytest.fixture
def open_fleet_ratelimited() -> OpenFleet:
    return kit.open_fleet(kit.fleet_of(kit.RateLimitedReader()))


@pytest.fixture
def open_fleet_overloaded() -> OpenFleet:
    return kit.open_fleet(kit.fleet_of(kit.OverloadedReader()))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(calibration=tmp_path / "calibration.json", model="m", env="dev")


@pytest.fixture
def request_systemone(request_bytes: bytes) -> SystemOneRequest:
    return msgspec.json.decode(request_bytes, type=SystemOneRequest)
