"""API fixtures: the real app over the fake reader, open and closed through its lifespan."""

from collections.abc import AsyncIterator

import pytest
from litestar.testing import AsyncTestClient

from e_jev.adapters.ports import OpenReader
from e_jev.core.settings import Settings
from e_jev.main import create_app


@pytest.fixture
async def client(settings: Settings, open_reader: OpenReader) -> AsyncIterator[AsyncTestClient]:
    async with AsyncTestClient(app=create_app(settings, open_reader)) as client:
        yield client


@pytest.fixture
async def client_down(settings: Settings, open_reader_down: OpenReader) -> AsyncIterator[AsyncTestClient]:
    async with AsyncTestClient(app=create_app(settings, open_reader_down)) as client:
        yield client


@pytest.fixture
async def client_failing(settings: Settings, open_reader_failing: OpenReader) -> AsyncIterator[AsyncTestClient]:
    async with AsyncTestClient(app=create_app(settings, open_reader_failing)) as client:
        yield client


@pytest.fixture
async def client_keyed(settings: Settings, open_reader: OpenReader) -> AsyncIterator[AsyncTestClient]:
    async with AsyncTestClient(app=create_app(settings.model_copy(update={"api_key": "k"}), open_reader)) as client:
        yield client
