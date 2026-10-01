import pytest
from litestar.testing import AsyncTestClient
from typesafe_sdk._core.response_types import ListModelsResponse, SystemOneResponse


async def test_systemone_evaluate_speaks_the_sdk_contract(client: AsyncTestClient, request_bytes: bytes) -> None:
    reply = await client.post("/v1/systemone", content=request_bytes, headers={"content-type": "application/json"})
    parsed = SystemOneResponse.model_validate_json(reply.content)
    assert (reply.status_code, sorted(parsed.answers)) == (200, ["department", "frustration", "is_urgent"])


@pytest.mark.parametrize(
    "body",
    [
        {"state": "s", "questions": {}},
        {"state": "s", "questions": {"q": {"type": "choice", "criteria": {"a": None}}}},
        {"state": "s", "questions": {"q": {"type": "rank", "instructions": "?"}}},
    ],
    ids=["no-questions", "one-option", "unknown-type"],
)
async def test_systemone_evaluate_invalid(client: AsyncTestClient, body: dict) -> None:
    reply = await client.post("/v1/systemone", json=body)
    assert (reply.status_code, reply.json()["detail"][0]["loc"]) == (422, ["body"])


async def test_systemone_models(client: AsyncTestClient) -> None:
    reply = await client.get("/v1/models")
    assert [model.name for model in ListModelsResponse.model_validate_json(reply.content).models] == ["jev-latest", "m"]


@pytest.mark.parametrize(("headers", "status"), [({}, 401), ({"authorization": "Bearer x"}, 401), ({"authorization": "Bearer k"}, 200)])
async def test_systemone_models_keyed(client_keyed: AsyncTestClient, headers: dict[str, str], status: int) -> None:
    assert (await client_keyed.get("/v1/models", headers=headers)).status_code == status


async def test_systemone_evaluate_upstream_failure(client_failing: AsyncTestClient, request_bytes: bytes) -> None:
    reply = await client_failing.post("/v1/systemone", content=request_bytes, headers={"content-type": "application/json"})
    assert (reply.status_code, reply.json()) == (502, {"detail": "readout refused"})


@pytest.mark.parametrize(("size", "status"), [(255, 200), (256, 422)])
async def test_systemone_evaluate_choice_limit(client: AsyncTestClient, size: int, status: int) -> None:
    body = {"state": "s", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {f"o{i}": None for i in range(size)}}}}
    reply = await client.post("/v1/systemone", json=body)
    assert reply.status_code == status


async def test_systemone_request_id_header(client: AsyncTestClient) -> None:
    ids = {(await client.get("/v1/models")).headers["x-typesafe-request-id"] for _ in range(3)}
    assert (len(ids), {len(value) for value in ids}) == (3, {32})
