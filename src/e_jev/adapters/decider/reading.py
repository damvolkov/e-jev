"""adapters.decider.reading: today's logprob readout behind the Decider port — the teacher, the fallback.

Every question re-sends the same state prefix, which the model server computes once (prefix cache).
With two permutations each question is asked in both option orders and the distributions averaged.
Scores stay raw and in log space: calibration is the engine's, never the backend's.
"""

import asyncio
from collections.abc import Mapping
from typing import Any, Final

import msgspec
import numpy as np
from beartype import beartype
from openinference.semconv.trace import OpenInferenceMimeTypeValues, OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from scipy.special import logsumexp

from e_jev.adapters.ports import Reader
from e_jev.config.backends import Backend
from e_jev.logic.labels import Node, Tokens, frontier_labels, normalize_labels, plan_labels, prune_labels, score_labels, select_labels
from e_jev.logic.primitives import Readout, orders_readout, prompt_readout, read_question
from e_jev.logic.usage import count_tokens
from e_jev.models.decision import Decision
from e_jev.models.reading import Reading
from e_jev.models.systemone import Json, Question

tracer = trace.get_tracer("e_jev")
KIND: Final = SpanAttributes.OPENINFERENCE_SPAN_KIND
JSON: Final = OpenInferenceMimeTypeValues.JSON.value


@beartype
class ReadingDecider:
    """The label readout over a `Reader`: one forward set of passes per question, averaged over orders."""

    __slots__ = ("_epsilon", "_model", "_permutations", "_reader", "_slots", "backend")

    def __init__(self, reader: Reader, spec: Backend, *, permutations: int, concurrency: int, epsilon: float = 1e-4) -> None:
        self.backend = spec.id
        self._reader = reader
        self._model = spec.model
        self._permutations = permutations
        self._slots = asyncio.Semaphore(concurrency)
        self._epsilon = epsilon

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

    async def _read_question(self, state: str, question: Question) -> Reading:
        """One question: the order-averaged log distribution over its options."""
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

    ############################################################

    ##### PUBLIC #####

    async def decide(self, state: str, questions: Mapping[str, Question]) -> Decision:
        """Every question read in parallel over the shared state: raw log scores, billed as the prefix cache computes."""
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(self._read_question(state, question)) for question in questions.values()]
        readings = [task.result() for task in tasks]
        return Decision(
            scores=dict(zip(questions, (reading.scores for reading in readings), strict=True)),
            input_tokens=count_tokens([prompt for reading in readings for prompt in reading.prompts]),
            output_tokens=sum(reading.calls for reading in readings),
        )

    @property
    def reader(self) -> Reader:
        """The underlying model server, kept reachable for the one capability no decision model has: extraction."""
        return self._reader

    async def extract(self, prompt: str, schema: dict[str, Any]) -> Json:
        """The teacher's own generation: no decision model answers this, the reading backend does."""
        return await self._reader.extract(prompt, schema)

    async def health(self) -> bool:
        return await self._reader.health()
