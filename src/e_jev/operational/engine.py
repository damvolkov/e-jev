"""operational.engine: System One over a fleet of Deciders — routing, calibration, answers.

One request fans out per question along its fallback chain (the row for its kind, `logic.dispatch`): the
questions each backend may answer go to it in one call; a backend that fails hands its questions to the
next in the row, marked in the trace — never silently. Calibration is e-jev's alone: one temperature per
(backend, kind); confidence and expectations are rebuilt from the raw scores, whatever the backend
published beside them.
"""

import asyncio
from collections.abc import Mapping, Sequence
from typing import Final

import msgspec
import structlog
from beartype import beartype
from openinference.semconv.trace import OpenInferenceMimeTypeValues, OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace

from e_jev.adapters.ports import Decider, Extractor
from e_jev.config.backends import Backend
from e_jev.core.errors import CapabilityError, JevError, RequestError, RetryableError
from e_jev.logic.calibration import scale_temperature
from e_jev.logic.dispatch import chains
from e_jev.logic.primitives import build_answer, render_json
from e_jev.models.calibration import Calibration
from e_jev.models.decision import Decision
from e_jev.models.extract import ExtractRequest
from e_jev.models.reading import Scores
from e_jev.models.systemone import Answer, Json, Question, QuestionKind, SystemOneRequest, SystemOneResponse, Usage

tracer = trace.get_tracer("e_jev")
log = structlog.get_logger()
KIND: Final = SpanAttributes.OPENINFERENCE_SPAN_KIND
JSON: Final = OpenInferenceMimeTypeValues.JSON.value

EXTRACT = "{instructions}\n\nState:\n{state}\n\nReply only with JSON matching the schema."


class EngineEvaluateError(RequestError):
    """Routed questions failed on every backend of their row; the distinct causes, in one error."""

    def __init__(self, failures: Sequence[Exception]) -> None:
        super().__init__("; ".join(sorted({self._message(error) for error in failures})))
        self.failures = list(failures)

    @staticmethod
    def _message(error: BaseException) -> str:
        match error:
            case JevError(message=message):
                return message
            case _:
                return str(error)


def _jev_leaves(error: BaseException) -> list[JevError]:
    """The JevError leaves of a possibly nested exception group: a wave's failures, flattened."""
    match error:
        case BaseExceptionGroup(exceptions=inner):
            return [leaf for nested in inner for leaf in _jev_leaves(nested)]
        case JevError() as error:
            return [error]
        case _:
            return []


def _capacity_or_group(errors: list[Exception]) -> Exception:
    """A whole fleet that ran out of capacity is answered as capacity, so the SDK retries; otherwise it is 502."""
    return (
        RetryableError.from_group(ExceptionGroup("fleet at capacity", errors))
        if all(isinstance(error, RetryableError) for error in errors)
        else EngineEvaluateError(errors)
    )


@beartype
class Engine:
    __slots__ = ("_calibration", "_deciders", "_extractor", "_model", "_rows", "_specs", "calibrated")

    def __init__(
        self,
        deciders: Mapping[str, Decider],
        *,
        model: str,
        rows: Mapping[QuestionKind, Sequence[str]] | None = None,
        specs: Sequence[Backend],
        calibration: Calibration | None,
    ) -> None:
        self._deciders = deciders
        self._model = model
        self._rows = rows or {}
        self._specs = {spec.id: spec for spec in specs}
        self._calibration = calibration
        self._extractor = next((decider for decider in deciders.values() if isinstance(decider, Extractor)), None)
        self.calibrated = calibration is not None

    ##### PRIVATE #####

    async def _decide(self, backend: str, state: str, subset: Mapping[str, Question], step: int) -> tuple[Decision | None, list[JevError]]:
        """One route call, traced; its failures return as data so siblings keep answering and the row can fall back."""
        with tracer.start_as_current_span(f"decide {backend}") as span:
            span.set_attributes(
                {
                    KIND: OpenInferenceSpanKindValues.LLM.value,
                    "jev.backend": backend,
                    "jev.route_step": step,
                    "jev.questions": len(subset),
                    SpanAttributes.INPUT_VALUE: msgspec.json.encode(dict(subset)).decode(),
                    SpanAttributes.INPUT_MIME_TYPE: JSON,
                }
            )
            decision, failures = await self._attempt(backend, state, subset)
            match decision:
                case None:
                    span.set_attribute("jev.failed", "; ".join(map(EngineEvaluateError._message, failures)))
                case ok:
                    span.set_attributes(
                        {
                            SpanAttributes.OUTPUT_VALUE: msgspec.json.encode(
                                {name: scores.tolist() for name, scores in ok.scores.items()}
                            ).decode(),
                            SpanAttributes.OUTPUT_MIME_TYPE: JSON,
                            "jev.input_tokens": ok.input_tokens,
                        }
                    )
            return decision, failures

    async def _attempt(self, backend: str, state: str, subset: Mapping[str, Question]) -> tuple[Decision | None, list[JevError]]:
        """One decider call: its JevError leaves, bare or wrapped by an inner TaskGroup, come back as data."""
        decision: Decision | None = None
        failures: list[JevError] = []
        try:
            decision = await self._deciders[backend].decide(state, subset)
        except* JevError as group:
            failures = _jev_leaves(group)
            log.warning("route_failed", backend=backend, error="; ".join(map(EngineEvaluateError._message, failures)))
        return decision, failures

    async def _route_wave(
        self,
        state: str,
        questions: Mapping[str, Question],
        *,
        pending: Mapping[str, Sequence[Backend]],
        results: dict[str, tuple[str, Scores]],
        errors: list[Exception],
        decisions: list[Decision],
        step: int,
    ) -> None:
        """Ask every pending question of the backend at the head of its chain; what fails descends to the next."""
        match len(pending):
            case 0:
                return
        heads = {chain[0].id: chain[0] for chain in pending.values()}
        groups = {
            backend_id: {name: question for name, question in questions.items() if pending[name][0].id == backend_id}
            for backend_id in heads
        }
        async with asyncio.TaskGroup() as group:
            tasks = {backend_id: group.create_task(self._decide(backend_id, state, subset, step)) for backend_id, subset in groups.items()}
        for backend_id, task in tasks.items():
            decision, failures = task.result()
            errors.extend(failures)
            match decision:
                case None:
                    continue
                case ok:
                    decisions.append(ok)
                    results.update({name: (backend_id, scores) for name, scores in ok.scores.items()})
        await self._route_wave(
            state,
            questions,
            pending={name: chain[1:] for name, chain in pending.items() if name not in results and len(chain) > 1},
            results=results,
            errors=errors,
            decisions=decisions,
            step=step + 1,
        )

    def _answer(self, name: str, question: Question, backend: str, scores: Scores) -> Answer:
        """One routed score vector: calibrated with its own backend's temperature, answered, traced."""
        with tracer.start_as_current_span(f"question {name}") as span:
            span.set_attributes(
                {
                    KIND: OpenInferenceSpanKindValues.CHAIN.value,
                    "jev.backend": backend,
                    SpanAttributes.INPUT_VALUE: msgspec.json.encode(question).decode(),
                    SpanAttributes.INPUT_MIME_TYPE: JSON,
                }
            )
            temperature = self._calibration.temperature(backend, QuestionKind.of(question)) if self._calibration else 1.0
            span.set_attribute("jev.temperature", temperature)
            answer = build_answer(question, scale_temperature(scores, temperature))
            span.set_attributes({SpanAttributes.OUTPUT_VALUE: msgspec.json.encode(answer).decode(), SpanAttributes.OUTPUT_MIME_TYPE: JSON})
        return answer

    ############################################################

    ##### PUBLIC #####

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
            route = chains(self._rows, self._specs, request.questions, state)
            results: dict[str, tuple[str, Scores]] = {}
            errors: list[Exception] = []
            decisions: list[Decision] = []
            await self._route_wave(state, request.questions, pending=route, results=results, errors=errors, decisions=decisions, step=0)
            match len(results) == len(request.questions):
                case False:
                    raise _capacity_or_group(errors)
            response = SystemOneResponse(
                model=self._model,
                answers={name: self._answer(name, question, *results[name]) for name, question in request.questions.items()},
                usage=Usage(
                    input_tokens=sum(decision.input_tokens for decision in decisions),
                    output_tokens=sum(decision.output_tokens for decision in decisions),
                ),
            )
            span.set_attributes(
                {SpanAttributes.OUTPUT_VALUE: msgspec.json.encode(response).decode(), SpanAttributes.OUTPUT_MIME_TYPE: JSON}
            )
        return response

    async def extract(self, request: ExtractRequest) -> Json:
        """Extraction belongs to the generative teacher: the one decider that kept a reader behind it."""
        match self._extractor:
            case None:
                message = "no reading backend in the fleet to extract with"
                raise CapabilityError(message)
            case reader:
                prompt = EXTRACT.format(instructions=render_json(request.instructions), state=render_json(request.state))
                return await reader.extract(prompt, request.schema)

    async def health(self) -> bool:
        """Ready while any backend answers: the fleet degrades, it does not all have to stand."""
        return any(await asyncio.gather(*(decider.health() for decider in self._deciders.values())))
