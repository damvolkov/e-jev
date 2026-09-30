"""Logger fixtures: global structlog state restored after every test."""

from collections.abc import Iterator

import pytest
import structlog


@pytest.fixture(autouse=True)
def structlog_reset() -> Iterator[None]:
    yield
    structlog.reset_defaults()
