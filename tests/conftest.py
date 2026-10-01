"""Session-wide fixtures: physical resources, and the real `ejev` app for every tier."""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from e_jev.cli.main import build

RESOURCES = Path(__file__).parent / "resources"


@pytest.fixture(scope="session")
def request_bytes() -> bytes:
    return (RESOURCES / "systemone_request.json").read_bytes()


@pytest.fixture
def ejev(capfd: pytest.CaptureFixture[str]) -> Callable[..., Awaitable[tuple[int, str, str]]]:
    """Run the real app on its own loop, in a thread: (exit code, stdout, stderr), exactly what a shell would see."""

    async def run(*tokens: str) -> tuple[int, str, str]:
        capfd.readouterr()
        code = await asyncio.to_thread(build().meta, list(tokens))
        out, err = capfd.readouterr()
        return code or 0, out, err

    return run
