from litestar.testing import AsyncTestClient


async def test_health_live(client: AsyncTestClient) -> None:
    assert (await client.get("/health")).json() == {"status": "ok"}


async def test_health_ready(client: AsyncTestClient) -> None:
    assert (await client.get("/ready")).status_code == 200


async def test_health_ready_reader_down(client_down: AsyncTestClient) -> None:
    assert (await client_down.get("/ready")).json() == {"status": "reader_down"}
