"""The System One engine over a running vLLM server.

Readout: one forward pass per (question, option order), no text generated. `logprob_token_ids`
returns the exact raw logprobs of the option letters; renormalizing them over the letters is the
softmax restricted to your options, so nothing outside them can win. `allowed_token_ids` pins the
single sampled token to a letter too. Every question re-sends the same state prefix, which vLLM's
prefix cache computes once. Typed extraction goes through Outlines as vLLM structured outputs.
"""

import asyncio
import string
from typing import NamedTuple, Self

import msgspec
import numpy as np
import outlines
import structlog
from httpx import AsyncClient
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChoiceLogprobs
from outlines.types import JsonSchema
from scipy.special import logsumexp

from e_jev.calibration import Scores, scale
from e_jev.models import Calibration, ExtractRequest, Json, Question, SystemOneRequest, SystemOneResponse, Usage
from e_jev.primitives import Readout, answer, readout, render
from e_jev.settings import Settings

log = structlog.get_logger()

LETTERS = string.ascii_uppercase
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
ASK = "{task} Reply with only the letter of your answer.\n\nState:\n{state}\n\nQuestion:\n{instructions}\n\nOptions:\n{options}"
EXTRACT = "{instructions}\n\nState:\n{state}\n\nReply only with JSON matching the schema."


class Reading(NamedTuple):
    scores: Scores
    input_tokens: int
    calls: int


def prompt(state: str, question: Readout, order: tuple[int, ...]) -> str:
    options = "\n".join(f"{letter}. {question.options[index]}" for letter, index in zip(LETTERS, order, strict=False))
    return ASK.format(task=question.task, state=state, instructions=question.instructions, options=options)


def orders(size: int, permutations: int) -> tuple[tuple[int, ...], ...]:
    """Option orders to ask: identity, then reversed — each position bias meets its mirror."""
    identity = tuple(range(size))
    return (identity, identity[::-1])[:permutations]


class Engine:
    __slots__ = ("calibration", "client", "extractor", "letters", "settings", "slots")

    def __init__(self, settings: Settings, client: AsyncOpenAI, letters: tuple[int, ...]) -> None:
        self.settings = settings
        self.client = client
        self.letters = letters
        self.slots = asyncio.Semaphore(settings.concurrency)
        self.extractor = outlines.from_vllm(client, settings.model)
        self.calibration = self.load(settings)

    @classmethod
    async def connect(cls, settings: Settings) -> Self:
        """Resolve the single-token id of every option letter from the served tokenizer."""
        client = AsyncOpenAI(base_url=settings.vllm_url, api_key=settings.vllm_api_key, max_retries=2)
        root = settings.vllm_url.removesuffix("/v1")
        headers = {"Authorization": f"Bearer {settings.vllm_api_key}"}
        async with AsyncClient(base_url=root, headers=headers, timeout=30) as http:
            replies = await asyncio.gather(
                *(
                    http.post("/tokenize", json={"model": settings.model, "prompt": letter, "add_special_tokens": False})
                    for letter in LETTERS
                )
            )
        tokens = tuple(reply.raise_for_status().json()["tokens"] for reply in replies)
        match [letter for letter, ids in zip(LETTERS, tokens, strict=True) if len(ids) != 1]:
            case [_, *_] as split:
                raise ValueError(f"letters not single-token for {settings.model}: {split}")
        return cls(settings, client, tuple(ids[0] for ids in tokens))

    @staticmethod
    def load(settings: Settings) -> Calibration | None:
        """Apply a fitted temperature only when it was fitted for this exact model and permutation count."""
        path = settings.calibration
        fitted = msgspec.json.decode(path.read_bytes(), type=Calibration) if path.is_file() else None
        match fitted:
            case Calibration(model=settings.model, permutations=settings.permutations):
                log.info("calibration.loaded", temperature=fitted.temperature, ece_after=fitted.ece_after)
                return fitted
            case Calibration():
                log.warning("calibration.stale", fitted=fitted.model, fitted_permutations=fitted.permutations)
                return None
            case None:
                log.warning("calibration.missing", path=str(path))
                return None

    async def logprobs(self, state: str, question: Readout, order: tuple[int, ...]) -> Reading:
        """Raw logprobs of the option letters for one order, returned in the original option order."""
        ids = list(self.letters[: len(order)])
        async with self.slots:
            reply = await self.client.chat.completions.create(
                model=self.settings.model,
                messages=[{"role": "user", "content": prompt(state, question, order)}],
                max_tokens=1,
                temperature=0,
                logprobs=True,
                extra_body={**NO_THINK, "logprob_token_ids": ids, "allowed_token_ids": ids, "return_tokens_as_token_ids": True},
            )
        match reply.choices[0].logprobs:
            case ChoiceLogprobs(content=[content, *_]):
                by_id = {int(entry.token.removeprefix("token_id:")): entry.logprob for entry in content.top_logprobs}
            case _:
                raise ValueError(f"vLLM returned no logprobs for {question.instructions!r}")
        shown = np.array([by_id[token] for token in ids])
        return Reading(shown[np.argsort(order)], reply.usage.prompt_tokens if reply.usage else 0, 1)

    async def read(self, state: str, question: Question) -> Reading:
        """Log of the order-averaged option distribution: the uncalibrated score vector."""
        shape = readout(question)
        async with asyncio.TaskGroup() as group:
            tasks = [
                group.create_task(self.logprobs(state, shape, order)) for order in orders(len(shape.options), self.settings.permutations)
            ]
        readings = [task.result() for task in tasks]
        normalized = np.stack([reading.scores - logsumexp(reading.scores) for reading in readings])
        return Reading(
            logsumexp(normalized, axis=0) - np.log(len(readings)),
            sum(reading.input_tokens for reading in readings),
            len(readings),
        )

    async def system_one(self, request: SystemOneRequest) -> SystemOneResponse:
        state = render(request.state)
        async with asyncio.TaskGroup() as group:
            tasks = {name: group.create_task(self.read(state, question)) for name, question in request.questions.items()}
        readings = {name: task.result() for name, task in tasks.items()}
        temperature = self.calibration.temperature if self.calibration else 1.0
        return SystemOneResponse(
            model=self.settings.model,
            answers={name: answer(request.questions[name], scale(reading.scores, temperature)) for name, reading in readings.items()},
            usage=Usage(
                input_tokens=sum(reading.input_tokens for reading in readings.values()),
                output_tokens=sum(reading.calls for reading in readings.values()),
            ),
        )

    async def extract(self, request: ExtractRequest) -> Json:
        raw = await self.extractor(
            EXTRACT.format(instructions=render(request.instructions), state=render(request.state)),
            JsonSchema(request.schema),
            max_tokens=self.settings.extract_max_tokens,
            temperature=0,
            extra_body=NO_THINK,
        )
        return msgspec.json.decode(raw)

    async def aclose(self) -> None:
        await self.client.close()
