from litestar.testing import AsyncTestClient


async def test_extract(client: AsyncTestClient) -> None:
    reply = await client.post("/v1/extract", json={"state": "s", "instructions": "i", "schema": {"type": "object"}})
    assert (reply.status_code, reply.json()) == (200, {"schema": ["type"]})
