"""adapters.fleet: one backend refusing to open degrades the fleet, it does not sink it."""

from contextlib import AsyncExitStack

import pytest

from e_jev.adapters.fleet import DownDecider, SystemOneDecider, open_decider, open_fleet
from e_jev.config.backends import READING, Backend, BackendKind
from e_jev.core.errors import RequestError
from e_jev.core.settings import Settings
from e_jev.models.systemone import Noul

DEAD = Backend(id="dead", kind=BackendKind.SYSTEMONE, url="http://127.0.0.1:45999")


async def test_down_backend_raises_through_the_row() -> None:
    down = DownDecider("dead", "connection refused")
    assert (down.backend, await down.health()) == ("dead", False)
    with pytest.raises(RequestError, match="unavailable"):
        await down.decide("s", {"q": Noul(instructions="?")})


async def test_open_fleet_survives_a_dead_backend() -> None:
    settings = Settings(model="m", env="dev", vllm_url="http://127.0.0.1:45999/v1", backends=(READING,))
    async with open_fleet(settings) as fleet:
        assert isinstance(fleet["reading"], DownDecider)
        assert await fleet["reading"].health() is False


async def test_open_decider_connects_a_reachable_server() -> None:
    async with AsyncExitStack() as stack:
        _, decider = await open_decider(
            stack, Backend(id="up", kind=BackendKind.SYSTEMONE, url="http://127.0.0.1:1"), Settings(model="m", env="dev"), concurrency=4
        )
    assert isinstance(decider, SystemOneDecider)
