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
from e_jev.adapters.ports import Decider
from e_jev.adapters.reader.vllm import VllmReader
from e_jev.config.backends import Backend, BackendKind
from e_jev.core.settings import Settings

log = structlog.get_logger()


async def open_decider(stack: AsyncExitStack, spec: Backend, settings: Settings) -> tuple[str, Decider]:
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
            return spec.id, await stack.enter_async_context(SystemOneDecider.open(spec, settings.concurrency))


@asynccontextmanager
async def open_fleet(settings: Settings) -> AsyncIterator[Mapping[str, Decider]]:
    """Every backend of the fleet connected at once; the whole fleet is refused if one of them is."""
    async with AsyncExitStack() as stack:
        fleet = dict(await asyncio.gather(*(open_decider(stack, spec, settings) for spec in settings.backends)))
        log.info("fleet_opened", backends=sorted(fleet))
        yield fleet
