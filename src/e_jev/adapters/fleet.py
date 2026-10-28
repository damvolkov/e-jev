"""adapters.fleet: the fleet the settings describe, opened as one — readers and native servers alike.

Every backend spec becomes a Decider under its id: the reading backend opens the vLLM reader (the one
that also extracts), the native and Laya servers share the `/v1/systemone` client. The stack here is
the fleet's lifetime: one enter per decider, all closed on any failure or exit."""

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Final

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


REOPEN_DELAY: Final = 30.0


class DownDecider:
    """A backend that refused to open: it fails every request (the row falls past it) until the server
    comes back, then reopens itself — re-opened at most once per `REOPEN_DELAY`, so a GPU route stopped
    by design taxes no decision beyond the delay it waits inside.

    One dead member does not sink the fleet — the teacher vLLM is routinely stopped while the decision
    model serves — and the reader returns while jev runs: no restart is owed to a backend that came back."""

    __slots__ = ("_delay", "_live", "_lock", "_next_attempt", "_reason", "_reopen", "backend")

    def __init__(
        self,
        backend: str,
        reason: str,
        reopen: Callable[[AsyncExitStack], Awaitable[tuple[str, Decider]]],
        *,
        delay: float = REOPEN_DELAY,
    ) -> None:
        self.backend = backend
        self._reason = reason
        self._reopen = reopen
        self._delay = delay
        self._live: tuple[Decider, AsyncExitStack] | None = None
        self._lock = asyncio.Lock()
        self._next_attempt = 0.0

    ##### PRIVATE #####

    async def _renew(self) -> Decider | None:
        """The live backend once the server returns; a failed attempt keeps its fresh diagnosis and waits out the delay."""
        async with self._lock:
            match self._live:
                case (decider, _):
                    return decider
                case None:
                    if time.monotonic() < self._next_attempt:
                        return None
                    self._next_attempt = time.monotonic() + self._delay
                    stack = AsyncExitStack()
                    try:
                        _, decider = await self._reopen(stack)
                    except (RequestError, OSError) as error:
                        await stack.aclose()
                        self._reason = str(error)[:200]
                        return None
                    self._live = (decider, stack)
                    log.info("backend_recovered", backend=self.backend)
                    return decider

    ############################################################

    ##### PUBLIC #####

    async def decide(self, state: str, questions: Mapping[str, Question]) -> Decision:
        """The reopened backend answers once it is back; until then the row falls past this failure."""
        match await self._renew():
            case None:
                message = f"backend {self.backend} unavailable: {self._reason}"
                raise RequestError(message)
            case live:
                return await live.decide(state, questions)

    async def health(self) -> bool:
        live = await self._renew()
        return live is not None and await live.health()

    async def aclose(self) -> None:
        """Release what the reopen acquired: the fleet's stack calls this on exit."""
        async with self._lock:
            match self._live:
                case (_, stack):
                    await stack.aclose()
                    self._live = None
                case None:
                    return


async def open_decider(stack: AsyncExitStack, spec: Backend, settings: Settings, *, concurrency: int | None = None) -> tuple[str, Decider]:
    match spec.kind:
        case BackendKind.READING:
            reader = await stack.enter_async_context(VllmReader.open(settings))
            return spec.id, ReadingDecider(
                reader,
                spec.model_copy(update={"model": spec.model or settings.model}),
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
                down = DownDecider(spec.id, str(error)[:200], lambda live: open_decider(live, spec, settings))
                stack.push_async_callback(down.aclose)
                return spec.id, down

        fleet = dict(await asyncio.gather(*(best_effort(spec) for spec in settings.backends)))
        log.info("fleet_opened", backends=sorted(fleet))
        yield fleet
