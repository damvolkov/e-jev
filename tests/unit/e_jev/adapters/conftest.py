"""Adapter-level fixtures: the fake vLLM server per test and the settings that bind the reader to it."""

import re
from collections.abc import Iterator

import pytest
from pytest_httpserver import HTTPServer
from tests import kit

from e_jev.core.settings import Settings


@pytest.fixture
def fake_vllm() -> Iterator[kit.FakeVllm]:
    server = HTTPServer(host="127.0.0.1")
    fake = kit.FakeVllm(server)
    server.expect_request(re.compile(".*")).respond_with_handler(fake.respond)
    server.start()
    try:
        yield fake
    finally:
        server.stop()


@pytest.fixture
def vllm_settings(fake_vllm: kit.FakeVllm) -> Settings:
    return Settings(model="m", env="dev", vllm_url=fake_vllm.url_for("/v1"))


@pytest.fixture
def dead_vllm_settings() -> Settings:
    return Settings(model="m", env="dev", vllm_url="http://127.0.0.1:45999/v1")
