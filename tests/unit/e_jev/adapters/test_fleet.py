"""adapters.fleet: one backend refusing to open degrades the fleet, it does not sink it — and the backend that returns is reopened, once per delay."""

from contextlib import AsyncExitStack
from typing import Final

import pytest
from tests import kit

from e_jev.adapters.fleet import DownDecider, SystemOneDecider, open_decider, open_fleet
from e_jev.adapters.ports import Decider
from e_jev.config.backends import READING, Backend, BackendKind
from e_jev.core.errors import RequestError
from e_jev.core.settings import Settings
from e_jev.models.systemone import Noul

DEAD = Backend(id="dead", kind=BackendKind.SYSTEMONE, url="http://127.0.0.1:45999")


async def test_down_backend_raises_through_the_row() -> None:
    async def refuse(stack: AsyncExitStack) -> tuple[str, Decider]:
        message = "connection refused"
        raise OSError(message)

    down = DownDecider("dead", "connection refused", refuse)
    assert down.backend == "dead"
    assert await down.health() is False
    with pytest.raises(RequestError, match="unavailable: connection refused"):
        await down.decide("s", {"q": Noul(instructions="?")})


async def test_down_backend_returns_when_the_server_comes_back() -> None:
    live = kit.FakeDecider("dead")

    async def reopen(stack: AsyncExitStack) -> tuple[str, Decider]:
        return live.backend, live

    down = DownDecider("dead", "connection refused", reopen, delay=0.0)
    assert await down.health() is True
    decision = await down.decide("s", {"q": Noul(instructions="?")})
    assert (decision.input_tokens, live.seen) == (10, [("q",)])


async def test_down_backend_reopens_at_most_once_per_delay() -> None:
    attempts: Final = []

    async def counting(stack: AsyncExitStack) -> tuple[str, Decider]:
        attempts.append(1)
        message = "connection refused"
        raise OSError(message)

    down = DownDecider("dead", "refused", counting)
    assert (await down.health(), await down.health(), len(attempts)) == (False, False, 1)


async def test_down_backend_keeps_the_freshest_diagnosis() -> None:
    reasons = iter(("HTTP 404: no model", "HTTP 503: busy"))

    async def reopening(stack: AsyncExitStack) -> tuple[str, Decider]:
        raise RequestError(next(reasons))

    down = DownDecider("dead", "initial", reopening, delay=0.0)
    assert await down.health() is False
    with pytest.raises(RequestError, match="HTTP 503: busy"):
        await down.decide("s", {"q": Noul(instructions="?")})


async def test_down_backend_closes_what_it_reopened() -> None:
    live = kit.FakeDecider("dead")
    closed = False

    async def marking() -> None:
        nonlocal closed
        closed = True

    async def reopening(stack: AsyncExitStack) -> tuple[str, Decider]:
        stack.push_async_callback(marking)
        return live.backend, live

    down = DownDecider("dead", "down", reopening)
    assert (await down.health(), closed) == (True, False)
    await down.aclose()
    assert closed is True


async def test_down_backend_reopens_the_reading_route(fake_vllm: kit.FakeVllm, vllm_settings: Settings) -> None:
    fake_vllm.up = True
    down = DownDecider("reading", "vllm tokenize: unreachable", lambda live: open_decider(live, READING, vllm_settings), delay=0.0)
    assert await down.health() is True


async def test_open_fleet_survives_a_dead_backend() -> None:
    settings = Settings(model="m", env="dev", vllm_url="http://127.0.0.1:45999/v1", backends=(READING,))
    async with open_fleet(settings) as fleet:
        assert isinstance(fleet["reading"], DownDecider)
        assert await fleet["reading"].health() is False


async def test_open_fleet_names_the_refusal_and_waits_its_delay(fake_vllm: kit.FakeVllm, vllm_settings: Settings) -> None:
    async with open_fleet(vllm_settings.model_copy(update={"backends": (READING,)})) as fleet:
        assert isinstance(fleet["reading"], DownDecider)
        with pytest.raises(RequestError, match=r"HTTP 404: The model `m` does not exist"):
            await fleet["reading"].decide("s", {"q": Noul(instructions="?")})
        fake_vllm.up = True
        assert await fleet["reading"].health() is False


async def test_open_decider_connects_a_reachable_server() -> None:
    async with AsyncExitStack() as stack:
        _, decider = await open_decider(
            stack, Backend(id="up", kind=BackendKind.SYSTEMONE, url="http://127.0.0.1:1"), Settings(model="m", env="dev"), concurrency=4
        )
    assert isinstance(decider, SystemOneDecider)
