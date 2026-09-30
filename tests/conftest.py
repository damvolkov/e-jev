"""Session-wide fixtures: physical resources."""

from pathlib import Path

import pytest

RESOURCES = Path(__file__).parent / "resources"


@pytest.fixture(scope="session")
def request_bytes() -> bytes:
    return (RESOURCES / "systemone_request.json").read_bytes()
