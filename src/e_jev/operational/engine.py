"""operational.engine: System One evaluation over any Reader — orders, averaging, calibration, answers.

Every question re-sends the same state prefix, which the model server computes once (prefix cache).
With two permutations each question is asked in both option orders and the distributions averaged.
"""

import asyncio

import numpy as np
from beartype import beartype
from scipy.special import logsumexp

from e_jev.adapters.ports import Reader
from e_jev.core.errors import JevError, RequestError
from e_jev.logic.calibration import scale_temperature
from e_jev.logic.primitives import Readout, build_answer, orders_readout, prompt_readout, read_question, render_json
from e_jev.models.calibration import Calibration
from e_jev.models.extract import ExtractRequest
from e_jev.models.reading import Reading
from e_jev.models.systemone import Json, Question, SystemOneRequest, SystemOneResponse, Usage

EXTRACT = "{instructions}\n\nState:\n{state}\n\nReply only with JSON matching the schema."


class EngineEvaluateError(RequestError):
    """Parallel readouts failed upstream; the distinct causes, in one error."""

    def __init__(self, failures: BaseExceptionGroup) -> None:
        super().__init__("; ".join(sorted(set(self._init_leaves(failures)))))
        self.failures = failures

    @staticmethod
    def _init_leaves(error: BaseException) -> list[str]:
        match error:
            case BaseExceptionGroup(exceptions=inner):
                return [message for nested in inner for message in EngineEvaluateError._init_leaves(nested)]
            case JevError(message=message):
                return [message]
            case _:
                return [str(error)]


@beartype
class Engine:
    __slots__ = ("_model", "_permutations", "_reader", "_slots", "_temperature", "calibrated")

    def __init__(self, reader: Reader, model: str, permutations: int, concurrency: int, calibration: Calibration | None) -> None:
        self._reader = reader
        self._model = model
        self._permutations = permutations
        self._slots = asyncio.Semaphore(concurrency)
        self._temperature = calibration.temperature if calibration else 1.0
        self.calibrated = calibration is not None

    ##### PRIVATE #####

    async def _read_order(self, state: str, readout: Readout, order: tuple[int, ...]) -> Reading:
        """One readout in one option order, scores mapped back to the original option order."""
        async with self._slots:
            reading = await self._reader.logprobs(prompt_readout(state, readout, order), len(order))
        return Reading(scores=reading.scores[np.argsort(order)], input_tokens=reading.input_tokens, calls=reading.calls)

    ############################################################

    ##### PUBLIC #####

    async def read(self, state: str, question: Question) -> Reading:
        """Log of the order-averaged option distribution: the uncalibrated score vector."""
        readout = read_question(question)
        async with asyncio.TaskGroup() as group:
            tasks = [
                group.create_task(self._read_order(state, readout, order))
                for order in orders_readout(len(readout.options), self._permutations)
            ]
        readings = [task.result() for task in tasks]
        normalized = np.stack([reading.scores - logsumexp(reading.scores) for reading in readings])
        return Reading(
            scores=logsumexp(normalized, axis=0) - np.log(len(readings)),
            input_tokens=sum(reading.input_tokens for reading in readings),
            calls=sum(reading.calls for reading in readings),
        )

    async def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        state = render_json(request.state)
        try:
            async with asyncio.TaskGroup() as group:
                tasks = {name: group.create_task(self.read(state, question)) for name, question in request.questions.items()}
        except* RequestError as failures:
            raise EngineEvaluateError(failures) from failures
        readings = {name: task.result() for name, task in tasks.items()}
        return SystemOneResponse(
            model=self._model,
            answers={
                name: build_answer(request.questions[name], scale_temperature(reading.scores, self._temperature))
                for name, reading in readings.items()
            },
            usage=Usage(
                input_tokens=sum(reading.input_tokens for reading in readings.values()),
                output_tokens=sum(reading.calls for reading in readings.values()),
            ),
        )

    async def extract(self, request: ExtractRequest) -> Json:
        prompt = EXTRACT.format(instructions=render_json(request.instructions), state=render_json(request.state))
        return await self._reader.extract(prompt, request.schema)

    async def health(self) -> bool:
        return await self._reader.health()
