"""adapters.fleet: the fleet the settings describe, opened as one — readers and native servers alike.

Every backend spec becomes a Decider under its id: the reading backend opens the vLLM reader (the one
that also extracts), the native and Laya servers share the `/v1/systemone` client. The stack here is
the fleet's lifetime: one enter per decider, all closed on any failure or exit."""

import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager

import structlog

from e_jev.adapters.decider.reading import ReadingDecider
from e_jev.adapters.decider.systemone import SystemOneDecider
from e_jev.adapters.ports import Decider, Decision
from e_jev.adapters.reader.vllm import VllmReader
from e_jev.config.backends import Backend, BackendKind
from e_jev.core.errors import RequestError
from e_jev.core.settings import Settings
from e_jev.models.systemone import Question

log = structlog.get_logger()


class DownDecider:
    """A backend that refused to open: it fails every request (the row falls past it) and reports down.

    One dead member does not sink the fleet — the teacher vLLM is routinely stopped while the decision
    model serves, and the reader returns when `make gpu-teach` brings it back."""

    __slots__ = ("_reason", "backend")

    def __init__(self, backend: str, reason: str) -> None:
        self.backend = backend
        self._reason = reason

    async def decide(self, state: str, questions: Mapping[str, Question]) -> Decision:  # noqa: ARG002 — port signature; a down backend answers nothing.
        message = f"backend {self.backend} unavailable: {self._reason}"
        raise RequestError(message)

    async def health(self) -> bool:
        return False


async def open_decider(stack: AsyncExitStack, spec: Backend, settings: Settings, *, concurrency: int | None = None) -> tuple[str, Decider]:
    match spec.kind:
        case BackendKind.READING:
            reader = await stack.enter_async_context(VllmReader.open(settings))
            return spec.id, ReadingDecider(
                reader,
                spec.model_copy(update={"model": settings.model}),
                permutations=settings.permutations,
                concurrency=settings.concurrency,
                epsilon=settings.trie_epsilon,
            )
        case BackendKind.SYSTEMONE | BackendKind.LAYA:
            return spec.id, await stack.enter_async_context(SystemOneDecider.open(spec, concurrency or settings.concurrency))


@asynccontextmanager
async def open_fleet(settings: Settings) -> AsyncIterator[Mapping[str, Decider]]:
    """Every backend of the fleet connected at once; one that refuses to open degrades to a down decider."""
    async with AsyncExitStack() as stack:

        async def best_effort(spec: Backend) -> tuple[str, Decider]:
            try:
                return await open_decider(stack, spec, settings)
            except (RequestError, OSError) as error:
                log.warning("backend_unavailable", backend=spec.id, error=str(error)[:200])
                return spec.id, DownDecider(spec.id, str(error)[:200])

        fleet = dict(await asyncio.gather(*(best_effort(spec) for spec in settings.backends)))
        log.info("fleet_opened", backends=sorted(fleet))
        yield fleet
