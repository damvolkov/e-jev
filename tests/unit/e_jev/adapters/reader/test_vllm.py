"""adapters.reader.vllm: the upstream failure tree — capacity as 429/529 kinds, everything else as 502.

openai 3.x transports on httpx2 while the reader's own client is httpx: each exception is built with its
own response, and both status channels are classified identically."""

import httpx
import httpx2
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
from tests import kit

from e_jev.adapters.reader.vllm import VllmReader, VllmReaderError, VllmStage
from e_jev.core.errors import OverloadedError, RateLimitedError
from e_jev.core.settings import Settings

OPENAI_REQUEST = httpx2.Request("POST", "http://vllm/v1/completions")
HTTPX_REQUEST = httpx.Request("POST", "http://vllm/tokenize")


def _openai_error(status: int, headers: dict[str, str] | None = None) -> APIStatusError:
    response = httpx2.Response(status, headers=headers or {}, request=OPENAI_REQUEST, content=b'{"error": "x"}')
    return APIStatusError("upstream", response=response, body=None)


def _httpx_error(status: int, headers: dict[str, str] | None = None) -> httpx.HTTPStatusError:
    response = httpx.Response(status, headers=headers or {}, request=HTTPX_REQUEST)
    return httpx.HTTPStatusError("upstream", request=HTTPX_REQUEST, response=response)


def _rate_limited(headers: dict[str, str]) -> RateLimitError:
    response = httpx2.Response(429, headers=headers, request=OPENAI_REQUEST, content=b'{"error": "x"}')
    return RateLimitError("slow", response=response, body=None)


@pytest.mark.parametrize(
    ("error", "expected", "retry_after"),
    [
        (_rate_limited({"retry-after": "7"}), RateLimitedError, 7.0),
        (APITimeoutError(request=OPENAI_REQUEST), OverloadedError, None),
        (_openai_error(503), OverloadedError, None),
        (_openai_error(400), VllmReaderError, None),
        (APIConnectionError(request=OPENAI_REQUEST), VllmReaderError, None),
        (_httpx_error(429, {"retry-after": "3"}), RateLimitedError, 3.0),
        (_httpx_error(503), OverloadedError, None),
        (_httpx_error(500), VllmReaderError, None),
    ],
    ids=["sdk-rate", "timeout", "sdk-503", "sdk-gateway", "sdk-connection", "httpx-rate", "httpx-503", "httpx-gateway"],
)
def test_vllm_classify_capacity(error: Exception, expected: type, retry_after: float | None) -> None:
    failure = VllmReader._open_classify(VllmStage.READOUT, error)
    assert (type(failure), getattr(failure, "retry_after", None)) == (expected, retry_after)


@pytest.mark.parametrize(
    ("status", "content", "expected"),
    [
        (404, b'{"error": {"message": "The model `clef` does not exist."}}', "HTTP 404: The model `clef` does not exist."),
        (500, b"boom", "HTTP 500: Internal Server Error"),
        (400, b'{"detail": "no"}', "HTTP 400: Bad Request"),
    ],
    ids=["model-missing", "not-json", "no-error-field"],
)
def test_vllm_classify_status_names_the_server_reason(status: int, content: bytes, expected: str) -> None:
    reply = httpx.Response(status, request=HTTPX_REQUEST, content=content)
    failure = VllmReader._open_classify(VllmStage.TOKENIZE, httpx.HTTPStatusError("upstream", request=HTTPX_REQUEST, response=reply))
    assert failure.message == f"vllm tokenize: {expected}"


async def test_vllm_open_reports_unreachable_without_a_server(dead_vllm_settings: Settings) -> None:
    with pytest.raises(VllmReaderError, match="unreachable at"):
        async with VllmReader.open(dead_vllm_settings):
            pass


async def test_vllm_open_names_the_refusal_of_a_live_server(fake_vllm: kit.FakeVllm, vllm_settings: Settings) -> None:
    with pytest.raises(VllmReaderError, match=r"HTTP 404: The model `m` does not exist") as raised:
        async with VllmReader.open(vllm_settings):
            pass
    assert "unreachable" not in raised.value.message
