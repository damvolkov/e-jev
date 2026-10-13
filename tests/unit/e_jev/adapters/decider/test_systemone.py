"""adapters.decider.systemone: the native `/v1/systemone` client — contract in, log scores out, capacity typed."""

import httpx
import msgspec
import numpy as np
import pytest

from e_jev.adapters.decider.systemone import SystemOneDecider
from e_jev.config.backends import Backend, BackendKind, Language
from e_jev.core.errors import OverloadedError, RateLimitedError, RequestError
from e_jev.models.systemone import Choice, Noul, Score

SPEC = Backend(id="native", kind=BackendKind.SYSTEMONE, url="http://backend", model="m-native", languages=(Language.EN,))
STATE = "s"
QUESTIONS = {
    "n": Noul(instructions="?"),
    "c": Choice(criteria={"a": None, "b": "second"}, instructions="?"),
    "w": Choice(criteria={str(index): None for index in range(30)}, instructions="?"),
    "s": Score(criteria=["low", "mid", "high"], instructions="?"),
}
TABLE = {
    "n": {"type": "noul", "noul": 0.8},
    "c": {"type": "choice", "choice": "a", "confidence": 0.6, "probabilities": {"a": 0.8, "b": 0.2}},
    "w": {"type": "choice", "choice": "7", "confidence": 0.5, "probabilities": {str(i): (0.9 if i == 7 else 0.1 / 29) for i in range(30)}},
    "s": {
        "type": "score",
        "score": 1.2,
        "confidence": 0.4,
        "legend": {"0": "low", "1": "mid", "2": "high"},
        "probabilities": {"0": 0.1, "1": 0.7, "2": 0.2},
    },
}


def _transport(
    calls: list[httpx.Request], status: int = 200, headers: dict[str, str] | None = None, body: bytes = b""
) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        match status:
            case 200:
                asked = msgspec.json.decode(request.content)["questions"]
                return httpx.Response(
                    200,
                    content=msgspec.json.encode(
                        {"model": "m-native", "answers": {n: TABLE[n] for n in asked}, "usage": {"input_tokens": 21, "output_tokens": 0}}
                    ),
                )
            case _:
                return httpx.Response(status, headers=headers or {}, content=body)

    return httpx.MockTransport(handle)


def _decider(transport: httpx.MockTransport) -> SystemOneDecider:
    return SystemOneDecider(SPEC, httpx.AsyncClient(transport=transport, base_url="http://backend"))


async def test_systemone_decider_asks_every_question_once() -> None:
    calls: list[httpx.Request] = []
    decision = await _decider(_transport(calls)).decide(STATE, QUESTIONS)
    assert sorted(decision.scores) == ["c", "n", "s", "w"]
    assert decision.input_tokens == 21
    assert [(len(calls), calls[0].url.path, msgspec.json.decode(calls[0].content)["model"])] == [(1, "/v1/systemone", "m-native")]


async def test_systemone_decider_reads_scores_in_the_question_order() -> None:
    decision = await _decider(_transport([])).decide(STATE, QUESTIONS)
    assert [list(score.shape) for score in decision.scores.values()] == [[2], [2], [30], [3]]
    n, c, w, s = (decision.scores[k] for k in "ncws")
    assert (float(np.argmax(c)), float(np.argmax(w))) == (0.0, 7.0)
    assert np.exp(n).tolist() == pytest.approx([0.8, 0.2], rel=1e-6)
    assert np.exp(s).tolist() == pytest.approx([0.1, 0.7, 0.2], rel=1e-6)


@pytest.mark.parametrize(
    ("status", "headers", "expected"),
    [(429, {"retry-after": "2"}, RateLimitedError), (503, None, OverloadedError), (400, None, RequestError), (500, None, RequestError)],
)
async def test_systemone_decider_classifies_capacity(status: int, headers: dict | None, expected: type[BaseException]) -> None:
    failing = _decider(_transport([], status=status, headers=headers, body=b'{"detail": "x"}'))
    with pytest.raises(expected) as caught:
        await failing.decide(STATE, {"n": Noul(instructions="?")})
    assert (type(caught.value), getattr(caught.value, "retry_after", None)) == (expected, 2.0 if status == 429 else None)


async def test_systemone_decider_rejects_answers_out_of_contract() -> None:
    broken = httpx.MockTransport(
        lambda r: httpx.Response(
            200, content=b'{"model":"m","answers":{"other":{"type":"noul","noul":0.5}},"usage":{"input_tokens":1,"output_tokens":0}}'
        )
    )
    with pytest.raises(RequestError, match="out of contract"):
        await _decider(broken).decide(STATE, {"n": Noul(instructions="?")})


async def test_systemone_decider_health_probes_backend() -> None:
    ok = httpx.MockTransport(lambda r: httpx.Response(200 if r.url.path == "/health" else 404))
    assert await _decider(ok).health() is True
    down = httpx.MockTransport(lambda r: (_ for _ in ()).throw(httpx.ConnectError("gone")))
    assert await _decider(down).health() is False
