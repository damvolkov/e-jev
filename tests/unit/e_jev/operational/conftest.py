"""Engine fixtures over the fake reader."""

import pytest

from e_jev.adapters.ports import Reader
from e_jev.models.calibration import Calibration
from e_jev.operational.engine import Engine


@pytest.fixture
def engine(reader: Reader) -> Engine:
    return Engine(reader, model="m", permutations=1, concurrency=4, calibration=None)


@pytest.fixture
def engine_mirrored(reader: Reader) -> Engine:
    return Engine(reader, model="m", permutations=2, concurrency=4, calibration=None)


@pytest.fixture
def engine_cooled(reader: Reader) -> Engine:
    fitted = Calibration(model="m", permutations=1, temperature=1e6, ece_before=0.2, ece_after=0.05, samples=100)
    return Engine(reader, model="m", permutations=1, concurrency=4, calibration=fitted)
