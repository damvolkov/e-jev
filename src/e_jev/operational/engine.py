"""operational.engine: System One evaluation over any Reader — orders, averaging, calibration, answers.

Every question re-sends the same state prefix, which the model server computes once (prefix cache).
With two permutations each question is asked in both option orders and the distributions averaged.
"""

import asyncio
from typing import Final

import msgspec
import numpy as np
from beartype import beartype
from openinference.semconv.trace import OpenInferenceMimeTypeValues, OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from scipy.special import logsumexp

from e_jev.adapters.ports import Reader
from e_jev.core.errors import JevError, RequestError
from e_jev.logic.calibration import scale_temperature
from e_jev.logic.labels import Node, Tokens, frontier_labels, normalize_labels, plan_labels, prune_labels, score_labels, select_labels
from e_jev.logic.primitives import Readout, build_answer, orders_readout, prompt_readout, read_question, render_json
from e_jev.logic.usage import count_tokens
from e_jev.models.calibration import Calibration
from e_jev.models.extract import ExtractRequest
from e_jev.models.reading import Reading
from e_jev.models.systemone import Answer, Json, Question, QuestionKind, SystemOneRequest, SystemOneResponse, Usage

tracer = trace.get_tracer("e_jev")
KIND: Final = SpanAttributes.OPENINFERENCE_SPAN_KIND
JSON: Final = OpenInferenceMimeTypeValues.JSON.value

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
    __slots__ = ("_epsilon", "_model", "_permutations", "_reader", "_slots", "_temperatures", "calibrated")

    def __init__(
        self, reader: Reader, *, model: str, permutations: int, concurrency: int, calibration: Calibration | None, epsilon: float = 1e-4
    ) -> None:
        self._epsilon = epsilon
        self._reader = reader
        self._model = model
        self._permutations = permutations
        self._slots = asyncio.Semaphore(concurrency)
        self._temperatures = calibration.temperatures if calibration else {}
        self.calibrated = calibration is not None

    ##### PRIVATE #####

    async def _read_node(self, prompt: tuple[int, ...], node: Node) -> dict[int, float]:
        with tracer.start_as_current_span("readout") as span:
            span.set_attributes(
                {
                    KIND: OpenInferenceSpanKindValues.LLM.value,
                    SpanAttributes.LLM_MODEL_NAME: self._model,
                    SpanAttributes.LLM_TOKEN_COUNT_PROMPT: len(prompt) + len(node.prefix),
                    "readout.depth": len(node.prefix),
                    "readout.candidates": len(node.candidates),
                }
            )
            async with self._slots:
                scores = await self._reader.logprobs((*prompt, *node.prefix), node.candidates)
            logprobs = dict(zip(node.candidates, scores.tolist(), strict=True))
            span.set_attributes(
                {
                    SpanAttributes.OUTPUT_VALUE: msgspec.json.encode(normalize_labels(logprobs)).decode(),
                    SpanAttributes.OUTPUT_MIME_TYPE: JSON,
                }
            )
        return logprobs

    async def _read_wave(
        self,
        prompt: tuple[int, ...],
        nodes: dict[Tokens, Node],
        frontier: dict[Tokens, float],
        budget: float,
        answers: dict[Tokens, dict[int, float]],
    ) -> dict[Tokens, dict[int, float]]:
        """Expand one level best-first, then recurse into the next; the trie depth (≤3 for 255 labels) bounds it."""
        kept, dropped = prune_labels(frontier, budget)
        async with asyncio.TaskGroup() as group:
            tasks = {prefix: group.create_task(self._read_node(prompt, nodes[prefix])) for prefix in kept}
        read = {prefix: normalize_labels(task.result()) for prefix, task in tasks.items()}
        below = frontier_labels(list(nodes.values()), {prefix: frontier[prefix] for prefix in kept}, read)
        merged = answers | read
        return await self._read_wave(prompt, nodes, below, budget - dropped, merged) if below else merged

    async def _read_order(self, state: str, readout: Readout, order: tuple[int, ...]) -> Reading:
        """One option order: the prompt encoded once, the trie read best-first, scores back in the original order."""
        async with self._slots:
            prompt = await self._reader.encode(prompt_readout(state, readout, order))
        sequences = [self._reader.label(label) for label in select_labels(len(order))]
        planned = plan_labels(sequences, self._reader.stop)
        nodes = {node.prefix: node for node in planned}
        ### The root goes alone first: it prefills the prompt into the prefix cache every deeper pass then hits.
        answers = await self._read_wave(prompt, nodes, {(): 1.0}, self._epsilon, {})
        scores = score_labels(sequences, self._reader.stop, planned, answers)
        return Reading(scores=scores[np.argsort(order)], prompts=(prompt,), calls=len(answers))

    async def _evaluate_question(self, name: str, state: str, question: Question) -> tuple[Answer, Reading]:
        """One question: its reading, calibrated into a typed answer, traced as one step."""
        with tracer.start_as_current_span(f"question {name}") as span:
            span.set_attributes(
                {
                    KIND: OpenInferenceSpanKindValues.CHAIN.value,
                    SpanAttributes.INPUT_VALUE: msgspec.json.encode(question).decode(),
                    SpanAttributes.INPUT_MIME_TYPE: JSON,
                }
            )
            reading = await self.read(state, question)
            temperature = self._temperatures.get(QuestionKind.of(question), 1.0)
            span.set_attribute("jev.temperature", temperature)
            answer = build_answer(question, scale_temperature(reading.scores, temperature))
            span.set_attributes({SpanAttributes.OUTPUT_VALUE: msgspec.json.encode(answer).decode(), SpanAttributes.OUTPUT_MIME_TYPE: JSON})
        return answer, reading

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
            prompts=tuple(prompt for reading in readings for prompt in reading.prompts),
            calls=sum(reading.calls for reading in readings),
        )

    async def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        with tracer.start_as_current_span("systemone") as span:
            span.set_attributes(
                {
                    KIND: OpenInferenceSpanKindValues.CHAIN.value,
                    SpanAttributes.INPUT_VALUE: msgspec.json.encode(request).decode(),
                    SpanAttributes.INPUT_MIME_TYPE: JSON,
                }
            )
            state = render_json(request.state)
            (lead, question), *rest = request.questions.items()
            try:
                ### The first question alone prefills the shared state; the others then reuse it in parallel.
                async with asyncio.TaskGroup() as group:
                    first = group.create_task(self._evaluate_question(lead, state, question))
                async with asyncio.TaskGroup() as group:
                    tasks = {name: group.create_task(self._evaluate_question(name, state, item)) for name, item in rest}
            except* RequestError as failures:
                raise EngineEvaluateError(failures) from failures
            results = {lead: first.result()} | {name: task.result() for name, task in tasks.items()}
            response = SystemOneResponse(
                model=self._model,
                answers={name: answer for name, (answer, _) in results.items()},
                usage=Usage(
                    input_tokens=count_tokens([prompt for _, reading in results.values() for prompt in reading.prompts]),
                    output_tokens=sum(reading.calls for _, reading in results.values()),
                ),
            )
            span.set_attributes(
                {SpanAttributes.OUTPUT_VALUE: msgspec.json.encode(response).decode(), SpanAttributes.OUTPUT_MIME_TYPE: JSON}
            )
        return response

    async def extract(self, request: ExtractRequest) -> Json:
        prompt = EXTRACT.format(instructions=render_json(request.instructions), state=render_json(request.state))
        return await self._reader.extract(prompt, request.schema)

    async def health(self) -> bool:
        return await self._reader.health()
