"""adapters.reader.vllm: the Reader port over a vLLM OpenAI-compatible server.

The chat template renders each prompt once through `/tokenize`; readouts then go to `/v1/completions`
with the prompt as token ids, so a trie node is the same prompt plus a few label tokens — computed once by
the prefix cache. `logprob_token_ids` returns the exact raw logprobs of the candidates; `allowed_token_ids`
pins the single sampled token to one of them. Extraction goes through Outlines, which vLLM turns into
structured outputs.
"""

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from enum import StrEnum, auto
from typing import Any, Final, Self

import httpx
import msgspec
import numpy as np
import outlines
import structlog
from beartype import beartype
from openai import APIError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError
from opentelemetry.instrumentation.utils import suppress_instrumentation
from outlines.types import JsonSchema

from e_jev.core.errors import OverloadedError, RateLimitedError, RequestError
from e_jev.core.settings import Settings
from e_jev.core.telemetry import inject_headers
from e_jev.logic.labels import UNIVERSE
from e_jev.models.reading import Scores
from e_jev.models.systemone import Json

log = structlog.get_logger()

NO_THINK: Final = {"chat_template_kwargs": {"enable_thinking": False}}
PROBE: Final = "Z"


class VllmStage(StrEnum):
    TOKENIZE = auto()
    READOUT = auto()
    EXTRACT = auto()


class VllmReaderError(RequestError):
    """vLLM failed or answered out of contract at one stage."""

    def __init__(self, stage: VllmStage, detail: str) -> None:
        super().__init__(f"vllm {stage}: {detail}")
        self.stage = stage
        self.detail = detail


@beartype
class VllmReader:
    __slots__ = ("_client", "_extract_tokens", "_extractor", "_http", "_labels", "_model", "_stop")

    def __init__(
        self, client: AsyncOpenAI, http: httpx.AsyncClient, labels: dict[str, tuple[int, ...]], stop: int, settings: Settings
    ) -> None:
        self._client = client
        self._http = http
        self._labels = labels
        self._stop = stop
        self._model = settings.model
        self._extract_tokens = settings.extract_max_tokens
        self._extractor = outlines.from_vllm(client, settings.model)

    ##### PRIVATE #####

    @staticmethod
    async def _open_tokenize(http: httpx.AsyncClient, body: dict[str, Any]) -> tuple[int, ...]:
        reply = await http.post("/tokenize", json=body)
        return tuple(reply.raise_for_status().json()["tokens"])

    @staticmethod
    def _open_retry_after(headers: Mapping[str, str]) -> float | None:
        """The upstream `Retry-After`, honoured when it is a plain number of seconds."""
        match (headers.get("retry-after") or "").strip():
            case raw if raw.isdigit():
                return float(raw)
            case _:
                return None

    @staticmethod
    def _open_classify(stage: VllmStage, error: Exception) -> RequestError:
        """The typed upstream failure: capacity answers 429/529, everything else is a gateway error.

        openai 3.x rides on httpx2 while the reader's own client is httpx — the SDK's 429 is known by its
        class, httpx's by its status pattern, and the hint is read off either response as a mapping."""
        match error:
            case RateLimitError(response=reply) | httpx.HTTPStatusError(response=httpx.Response(status_code=429) as reply):
                return RateLimitedError(f"vllm {stage}: rate limited", VllmReader._open_retry_after(reply.headers))
            case APIStatusError(status_code=503) | httpx.HTTPStatusError(response=httpx.Response(status_code=503)):
                return OverloadedError(f"vllm {stage}: server overloaded")
            case APITimeoutError() | httpx.TimeoutException():
                return OverloadedError(f"vllm {stage}: timed out")
            case APIError(message=message):
                return VllmReaderError(stage, message)
            case httpx.HTTPStatusError(response=response):
                return VllmReaderError(stage, f"HTTP {response.status_code}: {VllmReader._open_detail(response)}")
            case _:
                return VllmReaderError(stage, str(error))

    @staticmethod
    def _open_detail(response: httpx.Response) -> str:
        """The reason the server states in its body when it states one: a 404 naming a missing model is not an unreachable host."""
        try:
            body = response.json()
        except ValueError:
            body = None
        match body:
            case {"error": {"message": str() as message}} if message:
                return message
            case _:
                return response.reason_phrase or f"HTTP {response.status_code}"

    @classmethod
    async def _open_labels(cls, http: httpx.AsyncClient, model: str) -> dict[str, tuple[int, ...]]:
        """Every label the readout may show, tokenized once by the served tokenizer — untraced startup work."""
        with suppress_instrumentation():
            tokens = await asyncio.gather(
                *(cls._open_tokenize(http, {"model": model, "prompt": label, "add_special_tokens": False}) for label in UNIVERSE)
            )
        return dict(zip(UNIVERSE, tokens, strict=True))

    @classmethod
    async def _open_stop(cls, http: httpx.AsyncClient, model: str) -> int:
        """The token the chat template writes right after an answer — read from the template, never assumed."""
        probe = await cls._open_tokenize(http, {"model": model, "prompt": PROBE, "add_special_tokens": False})
        turn = await cls._open_tokenize(
            http,
            {"model": model, "messages": [{"role": "user", "content": "?"}, {"role": "assistant", "content": PROBE}], **NO_THINK},
        )
        match [index for index in range(len(turn) - len(probe)) if turn[index : index + len(probe)] == probe]:
            case [*_, last]:
                return turn[last + len(probe)]
            case _:
                raise VllmReaderError(VllmStage.TOKENIZE, "chat template never renders an assistant answer")

    ############################################################

    ##### PUBLIC #####

    @classmethod
    @asynccontextmanager
    async def open(cls, settings: Settings) -> AsyncIterator[Self]:
        """A connected reader, or none: every client opened here is closed on any failure or exit."""
        async with AsyncExitStack() as stack:
            client = await stack.enter_async_context(AsyncOpenAI(base_url=settings.vllm_url, api_key=settings.vllm_api_key, max_retries=2))
            http = await stack.enter_async_context(
                httpx.AsyncClient(
                    base_url=settings.vllm_url.removesuffix("/v1"),
                    headers={"Authorization": f"Bearer {settings.vllm_api_key}"},
                    timeout=30,
                )
            )
            try:
                labels, stop = await asyncio.gather(cls._open_labels(http, settings.model), cls._open_stop(http, settings.model))
            except httpx.HTTPStatusError as error:
                raise cls._open_classify(VllmStage.TOKENIZE, error) from error
            except httpx.HTTPError as error:
                raise VllmReaderError(VllmStage.TOKENIZE, f"unreachable at {settings.vllm_url}") from error
            log.info("reader_opened", model=settings.model, url=settings.vllm_url, stop=stop)
            yield cls(client, http, labels, stop, settings)

    @property
    def stop(self) -> int:
        return self._stop

    def label(self, text: str) -> tuple[int, ...]:
        return self._labels[text]

    async def encode(self, prompt: str) -> tuple[int, ...]:
        body = {"model": self._model, "messages": [{"role": "user", "content": prompt}], "add_generation_prompt": True, **NO_THINK}
        try:
            return await self._open_tokenize(self._http, body)
        except httpx.HTTPError as error:
            raise self._open_classify(VllmStage.TOKENIZE, error) from error

    async def logprobs(self, tokens: Sequence[int], candidates: Sequence[int]) -> Scores:
        ids = list(candidates)
        try:
            reply = await self._client.completions.create(
                model=self._model,
                prompt=list(tokens),
                max_tokens=1,
                temperature=0,
                logprobs=1,
                extra_body={"logprob_token_ids": ids, "allowed_token_ids": ids, "return_tokens_as_token_ids": True},
                extra_headers=inject_headers(),
            )
        except APIError as error:
            raise self._open_classify(VllmStage.READOUT, error) from error
        match reply.choices[0].logprobs:
            case None:
                raise VllmReaderError(VllmStage.READOUT, "no logprobs returned")
            case logprobs:
                by_id = {int(token.removeprefix("token_id:")): value for token, value in (logprobs.top_logprobs or [{}])[0].items()}
        return np.array([by_id[token] for token in ids])

    async def extract(self, prompt: str, schema: dict[str, Any]) -> Json:
        try:
            raw = await self._extractor(
                prompt,
                JsonSchema(schema),
                max_tokens=self._extract_tokens,
                temperature=0,
                extra_body=NO_THINK,
                extra_headers=inject_headers(),
            )
        except APIError as error:
            raise self._open_classify(VllmStage.EXTRACT, error) from error
        return msgspec.json.decode(raw)

    async def health(self) -> bool:
        try:
            reply = await self._http.get("/health")
        except httpx.HTTPError:
            return False
        return reply.is_success
