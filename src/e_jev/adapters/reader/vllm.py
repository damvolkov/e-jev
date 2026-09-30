"""adapters.reader.vllm: the Reader port over a vLLM OpenAI-compatible server.

`logprob_token_ids` returns the exact raw logprobs of the option letters — renormalized by the core,
that is the softmax restricted to the options, so nothing outside them can win. `allowed_token_ids`
pins the single sampled token to a letter too. Extraction goes through Outlines, which vLLM turns
into structured outputs.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from enum import StrEnum, auto
from typing import Any, Final, Self

import httpx
import msgspec
import numpy as np
import outlines
import structlog
from beartype import beartype
from openai import APIError, AsyncOpenAI
from openai.types.chat.chat_completion import ChoiceLogprobs
from outlines.types import JsonSchema

from e_jev.core.errors import RequestError
from e_jev.core.settings import Settings
from e_jev.logic.primitives import LETTERS
from e_jev.models.reading import Reading
from e_jev.models.systemone import Json

log = structlog.get_logger()

NO_THINK: Final = {"chat_template_kwargs": {"enable_thinking": False}}


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
    __slots__ = ("_client", "_extract_tokens", "_extractor", "_http", "_letters", "_model")

    def __init__(self, client: AsyncOpenAI, http: httpx.AsyncClient, letters: tuple[int, ...], settings: Settings) -> None:
        self._client = client
        self._http = http
        self._letters = letters
        self._model = settings.model
        self._extract_tokens = settings.extract_max_tokens
        self._extractor = outlines.from_vllm(client, settings.model)

    ##### PRIVATE #####

    @staticmethod
    async def _open_letters(http: httpx.AsyncClient, model: str) -> tuple[int, ...]:
        """The single-token id of every option letter, from the served tokenizer."""
        replies = await asyncio.gather(
            *(http.post("/tokenize", json={"model": model, "prompt": letter, "add_special_tokens": False}) for letter in LETTERS)
        )
        tokens = tuple(reply.raise_for_status().json()["tokens"] for reply in replies)
        match [letter for letter, ids in zip(LETTERS, tokens, strict=True) if len(ids) != 1]:
            case [_, *_] as split:
                raise VllmReaderError(VllmStage.TOKENIZE, f"letters not single-token for {model}: {split}")
        return tuple(ids[0] for ids in tokens)

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
                letters = await cls._open_letters(http, settings.model)
            except httpx.HTTPError as error:
                raise VllmReaderError(VllmStage.TOKENIZE, f"unreachable at {settings.vllm_url}") from error
            log.info("reader_opened", model=settings.model, url=settings.vllm_url)
            yield cls(client, http, letters, settings)

    async def logprobs(self, prompt: str, size: int) -> Reading:
        ids = list(self._letters[:size])
        try:
            reply = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1,
                temperature=0,
                logprobs=True,
                extra_body={**NO_THINK, "logprob_token_ids": ids, "allowed_token_ids": ids, "return_tokens_as_token_ids": True},
            )
        except APIError as error:
            raise VllmReaderError(VllmStage.READOUT, error.message) from error
        match reply.choices[0].logprobs:
            case ChoiceLogprobs(content=[content, *_]):
                by_id = {int(entry.token.removeprefix("token_id:")): entry.logprob for entry in content.top_logprobs}
            case _:
                raise VllmReaderError(VllmStage.READOUT, "no logprobs returned")
        return Reading(
            scores=np.array([by_id[token] for token in ids]),
            input_tokens=reply.usage.prompt_tokens if reply.usage else 0,
            calls=1,
        )

    async def extract(self, prompt: str, schema: dict[str, Any]) -> Json:
        try:
            raw = await self._extractor(prompt, JsonSchema(schema), max_tokens=self._extract_tokens, temperature=0, extra_body=NO_THINK)
        except APIError as error:
            raise VllmReaderError(VllmStage.EXTRACT, error.message) from error
        return msgspec.json.decode(raw)

    async def health(self) -> bool:
        try:
            reply = await self._http.get("/health")
        except httpx.HTTPError:
            return False
        return reply.is_success
