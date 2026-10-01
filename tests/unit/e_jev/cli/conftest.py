"""CLI fixtures: the cli resource files."""

from pathlib import Path

import pytest

RESOURCES = Path(__file__).parents[3] / "resources" / "cli"


@pytest.fixture
def cli_resources() -> Path:
    return RESOURCES
