"""API fixtures: the real app over the fake fleet, open and closed through its lifespan."""

from collections.abc import AsyncIterator

import pytest
from litestar.testing import AsyncTestClient

from e_jev.adapters.ports import OpenFleet
from e_jev.core.settings import Settings
from e_jev.main import create_app


@pytest.fixture
async def client(settings: Settings, open_reader_fleet: OpenFleet) -> AsyncIterator[AsyncTestClient]:
    async with AsyncTestClient(app=create_app(settings, open_reader_fleet)) as client:
        yield client


@pytest.fixture
async def client_down(settings: Settings, open_fleet_down: OpenFleet) -> AsyncIterator[AsyncTestClient]:
    async with AsyncTestClient(app=create_app(settings, open_fleet_down)) as client:
        yield client


@pytest.fixture
async def client_failing(settings: Settings, open_fleet_failing: OpenFleet) -> AsyncIterator[AsyncTestClient]:
    async with AsyncTestClient(app=create_app(settings, open_fleet_failing)) as client:
        yield client


@pytest.fixture
async def client_ratelimited(settings: Settings, open_fleet_ratelimited: OpenFleet) -> AsyncIterator[AsyncTestClient]:
    async with AsyncTestClient(app=create_app(settings, open_fleet_ratelimited)) as client:
        yield client


@pytest.fixture
async def client_overloaded(settings: Settings, open_fleet_overloaded: OpenFleet) -> AsyncIterator[AsyncTestClient]:
    async with AsyncTestClient(app=create_app(settings, open_fleet_overloaded)) as client:
        yield client


@pytest.fixture
async def client_keyed(settings: Settings, open_reader_fleet: OpenFleet) -> AsyncIterator[AsyncTestClient]:
    async with AsyncTestClient(app=create_app(settings.model_copy(update={"api_key": "k"}), open_reader_fleet)) as client:
        yield client
